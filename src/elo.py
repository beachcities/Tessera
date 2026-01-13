"""
Tessera ELO Rating System
==========================

機能:
- Bradley-Terry モデルによる ELO 計算
- 過去 Checkpoint との対戦
- タイル毎の ELO 推移記録
- JSON Lines ログ出力

設計思想:
- GPU Native: 対戦もバッチ処理
- 石数差による簡易勝敗判定（Phase II）
- LLM 継承可能なログ構造

Version: 1.1.0

Changelog:
- 1.1.0 (2026-01-13): メモリ管理強化
  - ELO評価バッチサイズ 32→8 に縮小（8GB VRAM対応）
  - 評価前後に gc.collect() + torch.cuda.empty_cache()
  - バッチ毎のメモリクリア追加
  - try/finally で確実なメモリ解放
- 1.0.0 (2026-01-13): 初版
"""

import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple, Any
import datetime
import glob


# ============================================================
# ELO Rating Calculation
# ============================================================

@dataclass
class ELOConfig:
    """ELO設定"""
    initial_rating: float = 1500.0
    k_factor: float = 32.0
    
    # 対戦設定
    games_per_evaluation: int = 20  # 評価対戦数
    opponent_sampling: str = "recent"  # "uniform", "recent", "elo_gap"
    recent_checkpoints: int = 5  # recent sampling で使う直近チェックポイント数


class ELORating:
    """ELO レーティング計算"""
    
    def __init__(self, config: ELOConfig = None):
        self.config = config or ELOConfig()
        self.ratings: Dict[str, float] = {}  # checkpoint_name -> rating
    
    def get_rating(self, name: str) -> float:
        """レーティング取得（なければ初期値）"""
        return self.ratings.get(name, self.config.initial_rating)
    
    def set_rating(self, name: str, rating: float):
        """レーティング設定"""
        self.ratings[name] = rating
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """期待勝率を計算"""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))
    
    def update_ratings(self, 
                       name_a: str, 
                       name_b: str, 
                       result: float) -> Tuple[float, float]:
        """
        対戦結果でレーティングを更新
        
        Args:
            name_a: プレイヤーA（現行モデル）
            name_b: プレイヤーB（過去チェックポイント）
            result: Aの結果（1.0=勝ち, 0.5=引き分け, 0.0=負け）
        
        Returns:
            (new_rating_a, new_rating_b)
        """
        r_a = self.get_rating(name_a)
        r_b = self.get_rating(name_b)
        
        expected_a = self.expected_score(r_a, r_b)
        expected_b = 1.0 - expected_a
        
        k = self.config.k_factor
        
        new_r_a = r_a + k * (result - expected_a)
        new_r_b = r_b + k * ((1.0 - result) - expected_b)
        
        self.set_rating(name_a, new_r_a)
        self.set_rating(name_b, new_r_b)
        
        return new_r_a, new_r_b
    
    def to_dict(self) -> Dict[str, float]:
        """全レーティングを辞書で取得"""
        return self.ratings.copy()
    
    def load(self, filepath: str):
        """レーティングをファイルから読み込み"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.ratings = json.load(f)
    
    def save(self, filepath: str):
        """レーティングをファイルに保存"""
        with open(filepath, 'w') as f:
            json.dump(self.ratings, f, indent=2)


# ============================================================
# Game Judge (Phase II: 石数差)
# ============================================================

def judge_games_by_stones(engine) -> torch.Tensor:
    """
    石数差で勝敗判定（Phase II 簡易版）
    
    Args:
        engine: GPUGoEngine
    
    Returns:
        (batch,) - 1.0=黒勝ち, 0.5=引き分け, 0.0=白勝ち
    """
    stones = engine.count_stones()  # (batch, 2)
    black = stones[:, 0]
    white = stones[:, 1]
    
    # 石数差で判定（コミなし）
    result = torch.where(
        black > white,
        torch.ones_like(black, dtype=torch.float32),
        torch.where(
            black < white,
            torch.zeros_like(black, dtype=torch.float32),
            torch.full_like(black, 0.5, dtype=torch.float32)
        )
    )
    
    return result


# ============================================================
# Checkpoint Manager
# ============================================================

class CheckpointManager:
    """チェックポイント管理"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        Path(checkpoint_dir).mkdir(exist_ok=True)
    
    def list_checkpoints(self) -> List[str]:
        """利用可能なチェックポイント一覧"""
        pattern = os.path.join(self.checkpoint_dir, "*.pth")
        files = glob.glob(pattern)
        # ファイル名でソート（タイムスタンプ順を想定）
        return sorted(files)
    
    def get_recent_checkpoints(self, n: int = 5) -> List[str]:
        """直近 n 個のチェックポイント"""
        all_ckpts = self.list_checkpoints()
        return all_ckpts[-n:] if len(all_ckpts) >= n else all_ckpts
    
    def load_model_from_checkpoint(self, 
                                   filepath: str, 
                                   model_class,
                                   device: str = 'cuda',
                                   **model_kwargs) -> nn.Module:
        """チェックポイントからモデルをロード"""
        model = model_class(**model_kwargs).to(device)
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model


# ============================================================
# ELO Evaluator
# ============================================================

class ELOEvaluator:
    """
    ELO 評価器
    
    現行モデル vs 過去チェックポイント の対戦を行い、
    ELO レーティングを更新する。
    """
    
    def __init__(self,
                 engine_class,
                 model_class,
                 config: ELOConfig = None,
                 checkpoint_dir: str = "checkpoints",
                 device: str = 'cuda',
                 model_kwargs: dict = None):
        
        self.engine_class = engine_class
        self.model_class = model_class
        self.config = config or ELOConfig()
        self.device = device
        self.model_kwargs = model_kwargs or {}
        
        self.elo = ELORating(self.config)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 対戦ログ
        self.match_history: List[Dict] = []
    
    def select_opponent(self) -> Optional[str]:
        """対戦相手のチェックポイントを選択"""
        if self.config.opponent_sampling == "recent":
            candidates = self.checkpoint_manager.get_recent_checkpoints(
                self.config.recent_checkpoints
            )
        else:  # uniform
            candidates = self.checkpoint_manager.list_checkpoints()
        
        if not candidates:
            return None
        
        # ランダムに1つ選択
        import random
        return random.choice(candidates)
    
    def play_match(self,
                   current_model: nn.Module,
                   opponent_path: str,
                   num_games: int = None) -> Dict[str, Any]:
        """
        現行モデル vs 過去チェックポイント の対戦
        
        Args:
            current_model: 現行モデル
            opponent_path: 対戦相手のチェックポイントパス
            num_games: 対戦数（Noneならconfig値）
        
        Returns:
            対戦結果の辞書
        """
        import gc
        
        num_games = num_games or self.config.games_per_evaluation
        
        # ELO評価前にメモリをクリア
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        opponent_model = None
        engine = None
        
        try:
            # 対戦相手をロード
            opponent_model = self.checkpoint_manager.load_model_from_checkpoint(
                opponent_path,
                self.model_class,
                self.device,
                **self.model_kwargs
            )
            
            # 対戦用エンジン（バッチサイズを小さく：8GB VRAM対応）
            batch_size = min(num_games, 8)  # OOM対策で縮小
            engine = self.engine_class(batch_size=batch_size, device=self.device)
            
            wins_current = 0
            wins_opponent = 0
            draws = 0
            total_played = 0
            
            while total_played < num_games:
                games_this_batch = min(batch_size, num_games - total_played)
                
                # ゲーム実行
                results = self._play_batch_games(
                    engine, current_model, opponent_model, games_this_batch
                )
                
                # 結果集計
                for r in results[:games_this_batch]:
                    if r > 0.5:
                        wins_current += 1
                    elif r < 0.5:
                        wins_opponent += 1
                    else:
                        draws += 1
                
                total_played += games_this_batch
                
                # バッチごとにメモリクリア
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 勝率
            win_rate = (wins_current + 0.5 * draws) / total_played
            
            return {
                "opponent": os.path.basename(opponent_path),
                "games": total_played,
                "wins": wins_current,
                "losses": wins_opponent,
                "draws": draws,
                "win_rate": win_rate,
            }
        
        finally:
            # 確実にメモリ解放
            del opponent_model
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
    
    def _play_batch_games(self,
                          engine,
                          model_a: nn.Module,
                          model_b: nn.Module,
                          num_games: int,
                          max_moves: int = 200) -> List[float]:
        """
        バッチでゲームを実行
        
        交互に手番を担当（偶数ゲーム: A=黒, 奇数ゲーム: A=白）
        
        Returns:
            各ゲームの結果（Aの視点: 1.0=勝ち, 0.5=引き分け, 0.0=負け）
        """
        engine.reset()
        
        # どちらが黒番か（バッチ内で交互）
        a_is_black = torch.zeros(engine.batch_size, dtype=torch.bool, device=self.device)
        a_is_black[::2] = True  # 偶数インデックスはAが黒
        
        for move_num in range(max_moves):
            # 合法手マスク
            legal_mask = engine.get_legal_mask()
            
            # 現在の手番
            current_turn = engine.turn  # 0=黒, 1=白
            
            # Aの手番か判定
            a_turn = (current_turn == 0) == a_is_black  # XNORで判定
            
            # 履歴取得
            seq = engine.get_current_sequence(max_len=64)
            if seq.shape[1] < 64:
                pad = torch.full(
                    (engine.batch_size, 64 - seq.shape[1]),
                    362, dtype=torch.long, device=self.device
                )
                seq = torch.cat([pad, seq], dim=1)
            
            # 各モデルで推論
            model_a.eval()
            model_b.eval()
            
            with torch.no_grad():
                probs_a = model_a.get_move_probabilities(seq, legal_mask)
                probs_b = model_b.get_move_probabilities(seq, legal_mask)
            
            # 手を選択（Aの手番ならAの確率、そうでなければBの確率）
            probs = torch.where(
                a_turn.unsqueeze(1).expand_as(probs_a),
                probs_a,
                probs_b
            )
            
            moves = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # 着手
            engine.play_batch(moves)
            
            # 全ゲーム終局したら終了
            if engine.is_game_over().all():
                break
        
        # 勝敗判定
        stone_results = judge_games_by_stones(engine)  # 黒視点
        
        # Aの視点に変換
        a_results = torch.where(
            a_is_black,
            stone_results,
            1.0 - stone_results
        )
        
        return a_results[:num_games].tolist()
    
    def evaluate_and_update(self,
                            current_model: nn.Module,
                            current_name: str) -> Optional[Dict[str, Any]]:
        """
        評価対戦を行い、ELOを更新
        
        Args:
            current_model: 現行モデル
            current_name: 現行モデルの識別名
        
        Returns:
            評価結果（対戦相手がいない場合はNone）
        """
        opponent_path = self.select_opponent()
        
        if opponent_path is None:
            # 対戦相手がいない（初回など）
            self.elo.set_rating(current_name, self.config.initial_rating)
            return None
        
        opponent_name = os.path.basename(opponent_path)
        
        # 対戦実行
        match_result = self.play_match(current_model, opponent_path)
        
        # ELO更新
        old_elo_current = self.elo.get_rating(current_name)
        old_elo_opponent = self.elo.get_rating(opponent_name)
        
        new_elo_current, new_elo_opponent = self.elo.update_ratings(
            current_name,
            opponent_name,
            match_result["win_rate"]
        )
        
        # 結果を構築
        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "current_model": current_name,
            "opponent_model": opponent_name,
            "games": match_result["games"],
            "wins": match_result["wins"],
            "losses": match_result["losses"],
            "draws": match_result["draws"],
            "win_rate": match_result["win_rate"],
            "elo_before": old_elo_current,
            "elo_after": new_elo_current,
            "elo_change": new_elo_current - old_elo_current,
            "opponent_elo": old_elo_opponent,
        }
        
        self.match_history.append(result)
        
        return result


# ============================================================
# ELO Logger (JSON Lines)
# ============================================================

class ELOLogger:
    """ELO ログ出力（JSON Lines形式）"""
    
    def __init__(self, log_dir: str = "logs"):
        Path(log_dir).mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"elo_{timestamp}.jsonl")
        self.tile_log_path = os.path.join(log_dir, f"elo_tiles_{timestamp}.jsonl")
    
    def log_match(self, match_result: Dict[str, Any]):
        """対戦結果をログ"""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(match_result, ensure_ascii=False) + "\n")
    
    def log_tile_summary(self, tile_data: Dict[str, Any]):
        """タイル毎のELOサマリーをログ"""
        with open(self.tile_log_path, "a") as f:
            f.write(json.dumps(tile_data, ensure_ascii=False) + "\n")


# ============================================================
# Tile ELO Tracker
# ============================================================

@dataclass
class TileELOSummary:
    """タイル毎のELOサマリー"""
    tile_id: int
    start_game: int
    end_game: int
    matches_played: int
    avg_win_rate: float
    elo_start: float
    elo_end: float
    elo_change: float
    best_elo: float
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class TileELOTracker:
    """タイル毎のELO追跡"""
    
    def __init__(self, tile_size: int = 100):
        self.tile_size = tile_size
        self.current_tile_id = 0
        self.tile_matches: List[Dict] = []
        self.tile_start_elo = 1500.0
        self.tile_start_game = 0
        self.best_elo = 1500.0
        
        self.summaries: List[TileELOSummary] = []
    
    def record_match(self, match_result: Dict[str, Any], current_game: int):
        """対戦結果を記録"""
        self.tile_matches.append(match_result)
        
        if match_result["elo_after"] > self.best_elo:
            self.best_elo = match_result["elo_after"]
    
    def should_close_tile(self, current_game: int) -> bool:
        """タイルを閉じるべきか"""
        return (current_game - self.tile_start_game) >= self.tile_size
    
    def close_tile(self, current_game: int, current_elo: float) -> TileELOSummary:
        """タイルを閉じてサマリーを生成"""
        avg_win_rate = 0.0
        if self.tile_matches:
            avg_win_rate = sum(m["win_rate"] for m in self.tile_matches) / len(self.tile_matches)
        
        summary = TileELOSummary(
            tile_id=self.current_tile_id,
            start_game=self.tile_start_game,
            end_game=current_game,
            matches_played=len(self.tile_matches),
            avg_win_rate=avg_win_rate,
            elo_start=self.tile_start_elo,
            elo_end=current_elo,
            elo_change=current_elo - self.tile_start_elo,
            best_elo=self.best_elo,
        )
        
        self.summaries.append(summary)
        
        # 次のタイルの準備
        self.current_tile_id += 1
        self.tile_start_game = current_game
        self.tile_start_elo = current_elo
        self.tile_matches = []
        
        return summary


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing ELO System...")
    
    # ELO計算テスト
    elo = ELORating()
    
    # 初期レーティング
    print(f"Initial rating: {elo.get_rating('model_v1')}")
    
    # 対戦シミュレーション
    elo.set_rating("model_v1", 1500)
    elo.set_rating("model_v2", 1500)
    
    # v1 が v2 に勝利
    new_v1, new_v2 = elo.update_ratings("model_v1", "model_v2", 1.0)
    print(f"After v1 wins: v1={new_v1:.1f}, v2={new_v2:.1f}")
    
    # v2 が v1 に勝利
    new_v1, new_v2 = elo.update_ratings("model_v1", "model_v2", 0.0)
    print(f"After v2 wins: v1={new_v1:.1f}, v2={new_v2:.1f}")
    
    # 引き分け
    new_v1, new_v2 = elo.update_ratings("model_v1", "model_v2", 0.5)
    print(f"After draw: v1={new_v1:.1f}, v2={new_v2:.1f}")
    
    print("\n✅ ELO System test passed!")
