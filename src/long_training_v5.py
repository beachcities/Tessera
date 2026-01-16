"""
Tessera Long Training Script v4.1.0
====================================
並列化 + ELO評価 + Graceful Shutdown 統合版
Clean Room Protocol 実装

Core Changes from v3:
- 各バッチが独立したゲームとして進行
- 終局した盤面は即座にリセット＆学習
- タイル毎にELO評価を実行
- 正確なGraceful Shutdown/Resume

Architecture:
- BATCH_SIZE = 同時進行ゲーム数
- 終局検知 → 学習 → リセット → 継続
- SSM発散リスクの低減（ゲーム毎にリセット）

Changelog:
- 4.1.0 (2026-01-13): Clean Room Protocol 実装
  - run_elo_evaluation で学習用リソースを一時解放
  - game_histories の明示的クリア（ゾンビVRAM対策）
  - エンジン再構築パイプライン
  - elo.py v1.3 との連携
- 4.0.2 (2026-01-13): チェックポイント保存のACD対応
  - A (Atomicity): 一時ファイル→renameでアトミック保存
  - C (Consistency): 保存後に読み込み検証
  - D (Durability): fsyncで確実にディスク書き込み
- 4.0.1 (2026-01-13): ELO評価時のメモリ管理強化
- 4.0.0 (2026-01-13): 初版

Version: 4.1.0
"""

import os
import sys
import json
import signal
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import traceback
import gc
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Dict, Any

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor import TesseraMonitor, defragment_gpu_memory
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, VOCAB_SIZE, PAD_TOKEN, BOARD_SIZE
from tessera_model import TesseraModel
from model import MambaStateCapture  # 互換用
from elo import ELOEvaluator, ELOLogger, TileELOTracker, ELOConfig, judge_games_by_stones


# ============================================================
# GPUGoEngine 拡張: reset_selected
# ============================================================

def add_reset_selected_to_engine():
    """GPUGoEngineにreset_selectedメソッドを動的に追加"""
    
    def reset_selected(self, mask: torch.Tensor):
        """
        指定された盤面のみリセット
        
        Args:
            mask: (batch,) - True=リセット対象
        """
        if not mask.any():
            return
        
        indices = mask.nonzero().squeeze(-1)
        
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)
        
        self.boards[indices] = 0
        self.turn[indices] = 0
        self.ko_point[indices] = -1
        self.last_move[indices] = -1
        self.consecutive_passes[indices] = 0
        self.move_count[indices] = 0
        self.history[indices] = PAD_TOKEN
    
    GPUGoEngine.reset_selected = reset_selected

# メソッドを追加
add_reset_selected_to_engine()


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    """設定"""
    
    # === 基本設定 ===
    NUM_GAMES: int = 1000000          # 総ゲーム数
    MAX_MOVES_PER_GAME: int = 200    # 1ゲームあたりの最大手数
    SEQ_LEN: int = 64                # シーケンス長
    
    # === バッチ設定（並列ゲーム数） ===
    BATCH_SIZE: int = 64             # 同時進行ゲーム数
    
    # === タイル設定 ===
    TILE_SIZE: int = 100             # 1タイル = 100ゲーム
    
    # === モデル設定 ===
    D_MODEL: int = 256
    N_LAYERS: int = 4
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    GRADIENT_CLIP_NORM: float = 0.5
    
    # === ELO設定 ===
    ELO_EVAL_INTERVAL: int = 100     # ELO評価間隔（ゲーム数）
    ELO_GAMES_PER_EVAL: int = 20     # 評価対戦数
    
    # === ログ・保存設定 ===
    LOG_INTERVAL: int = 50           # ログ出力間隔（ゲーム数）
    CHECKPOINT_INTERVAL: int = 500   # チェックポイント間隔
    
    # === 安全装置 ===
    SSM_NORM_LIMIT: float = 300.0
    MAX_VRAM_RATIO: float = 0.90
    
    # === 温度スケジューリング ===
    TEMP_START: float = 1.5
    TEMP_END: float = 0.8
    TEMP_DECAY_GAMES: int = 20000
    
    # === パス ===
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    
    @classmethod
    def auto_scale(cls, vram_gb: float) -> 'Config':
        """VRAM容量に応じて自動スケール"""
        config = cls()
        
        if vram_gb >= 70:      # A100 80GB
            config.BATCH_SIZE = 256
            config.D_MODEL = 512
            config.N_LAYERS = 8
        elif vram_gb >= 40:    # A100 40GB
            config.BATCH_SIZE = 128
            config.D_MODEL = 384
            config.N_LAYERS = 6
        elif vram_gb >= 20:    # RTX 3090/4090
            config.BATCH_SIZE = 128
            config.D_MODEL = 256
            config.N_LAYERS = 4
        elif vram_gb >= 10:    # RTX 3080
            config.BATCH_SIZE = 128
            config.D_MODEL = 256
            config.N_LAYERS = 4
        else:                  # RTX 4070 Laptop (8GB)
            config.BATCH_SIZE = 128
            config.D_MODEL = 256
            config.N_LAYERS = 4
        
        return config


# ============================================================
# Graceful Shutdown Handler
# ============================================================

class GracefulShutdown:
    """Graceful Shutdown ハンドラ"""
    
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        print("\n⚠️ Shutdown requested, finishing current batch...")
        self.shutdown_requested = True
    
    def should_stop(self) -> bool:
        return self.shutdown_requested


# ============================================================
# Logger
# ============================================================

class Logger:
    """デュアルログ"""
    
    def __init__(self, log_dir: str):
        Path(log_dir).mkdir(exist_ok=True)
        timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d_%H%M%S")
        
        self.console_log = os.path.join(log_dir, f"training_v4_{timestamp}.log")
        self.jsonl_log = os.path.join(log_dir, f"training_v4_{timestamp}.jsonl")
    
    def log(self, message: str, also_print: bool = True):
        timestamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        
        if also_print:
            print(line, flush=True)
        
        with open(self.console_log, "a") as f:
            f.write(line + "\n")
    
    def log_json(self, event_type: str, data: Dict[str, Any]):
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),
            "event": event_type,
            **data
        }
        with open(self.jsonl_log, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ============================================================
# Checkpoint Functions (ACD: Atomic, Consistent, Durable)
# ============================================================

def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler,
                    stats: Dict,
                    config: Config,
                    filepath: str,
                    verify: bool = True):
    """
    アトミック + 検証 + 永続化 によるチェックポイント保存
    
    A (Atomicity): 一時ファイル → rename でアトミック性を担保
    C (Consistency): 保存後に読み込み検証
    D (Durability): fsync で確実にディスクに書き込み
    
    Args:
        model: モデル
        optimizer: オプティマイザ
        scheduler: スケジューラ
        stats: 統計情報
        config: 設定
        filepath: 保存先パス
        verify: 保存後に検証するか（デフォルトTrue）
    """
    import tempfile
    import shutil
    
    dir_name = os.path.dirname(filepath) or '.'
    tmp_path = None
    
    try:
        # 一時ファイルを作成
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.pth.tmp')
        os.close(fd)
        
        # データを準備
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'stats': stats,
            'config': asdict(config),
        }
        
        # 一時ファイルに書き込み
        torch.save(checkpoint_data, tmp_path)
        
        # D (Durability): fsync で永続化
        with open(tmp_path, 'rb') as f:
            os.fsync(f.fileno())
        
        # A (Atomicity): アトミックにリネーム
        shutil.move(tmp_path, filepath)
        tmp_path = None  # 成功したのでクリーンアップ不要
        
        # ディレクトリも fsync（メタデータの永続化）
        try:
            dir_fd = os.open(dir_name, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass  # 一部の環境では失敗するが致命的ではない
        
        # C (Consistency): 検証
        if verify:
            try:
                loaded = torch.load(filepath, map_location='cpu', weights_only=False)
                # キーの存在確認
                assert 'model_state_dict' in loaded, "Missing model_state_dict"
                assert 'stats' in loaded, "Missing stats"
            except Exception as e:
                # 検証失敗 → ファイルを削除
                if os.path.exists(filepath):
                    os.remove(filepath)
                raise RuntimeError(f"Checkpoint verification failed: {e}")
        
    except Exception as e:
        # 失敗したら一時ファイルをクリーンアップ
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise RuntimeError(f"Checkpoint save failed: {e}")


def load_checkpoint(filepath: str,
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler,
                    device: str) -> Dict:
    """チェックポイントから復元"""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('stats', {})


# ============================================================
# Parallel Training Loop
# ============================================================

class ParallelTrainer:
    """並列学習トレーナー"""
    
    def __init__(self,
                 config: Config,
                 device: str = 'cuda'):
        
        self.config = config
        self.device = device
        
        # コンポーネント初期化
        self.engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
        self.model = TesseraModel(
            vocab_size=VOCAB_SIZE,
            d_model=config.D_MODEL,
            n_layers=config.N_LAYERS
        ).to(device)
        
        
        # Phase III: 段階的解凍（Progressive Unfreezing）
        # Mamba パラメータと新規層パラメータを分離
        mamba_params = [p for n, p in self.model.named_parameters() if n.startswith('mamba.')]
        new_layer_params = [p for n, p in self.model.named_parameters() if not n.startswith('mamba.')]
        
        # Phase 3.1: Mamba フリーズ（requires_grad=False）
        for p in mamba_params:
            p.requires_grad = False
        
        self.optimizer = optim.AdamW([
            {'params': mamba_params, 'lr': 0.0},  # Phase 3.1 では学習しない
            {'params': new_layer_params, 'lr': config.LEARNING_RATE}
        ], weight_decay=config.WEIGHT_DECAY)
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=1e-6
        )
        
        self.monitor = TesseraMonitor()
        #self.state_capture = MambaStateCapture(self.model)  # DISABLED: Memory leak
        
        # ELO評価器
        self.elo_evaluator = ELOEvaluator(
            engine_class=GPUGoEngine,
            model_class=TesseraModel,
            config=ELOConfig(games_per_evaluation=config.ELO_GAMES_PER_EVAL),
            checkpoint_dir=config.CHECKPOINT_DIR,
            device=device,
            model_kwargs={
                'vocab_size': VOCAB_SIZE,
                'd_model': config.D_MODEL,
                'n_layers': config.N_LAYERS
            }
        )
        self.elo_logger = ELOLogger(config.LOG_DIR)
        self.tile_elo_tracker = TileELOTracker(config.TILE_SIZE)
        
        # 各ゲームの履歴を保持
        self.game_histories: List[List[torch.Tensor]] = [[] for _ in range(config.BATCH_SIZE)]
        
        # 統計
        self.stats = {
            'total_games': 0,
            'total_moves': 0,
            'total_loss': 0.0,
            'loss_count': 0,
            'best_loss': float('inf'),
            'current_elo': 1500.0,
            'best_elo': 1500.0,
        }
        
        # 各ゲームの手数を追跡
        self.game_move_counts = torch.zeros(config.BATCH_SIZE, dtype=torch.long, device=device)
    
    def get_temperature(self, total_games: int) -> float:
        """温度を取得"""
        progress = min(1.0, total_games / self.config.TEMP_DECAY_GAMES)
        return self.config.TEMP_START - (self.config.TEMP_START - self.config.TEMP_END) * progress
    
    def step(self) -> Tuple[int, float]:
        """
        1ステップ実行（全盤面に1手着手）
        
        Returns:
            (finished_games, loss): 終局したゲーム数とロス
        """
        # 合法手マスク
        legal_mask = self.engine.get_legal_mask()
        
        # シーケンス準備
        seq = self.engine.get_current_sequence(max_len=self.config.SEQ_LEN)
        if seq.shape[1] < self.config.SEQ_LEN:
            pad = torch.full(
                (self.config.BATCH_SIZE, self.config.SEQ_LEN - seq.shape[1]),
                PAD_TOKEN, dtype=torch.long, device=self.device
            )
            seq = torch.cat([pad, seq], dim=1)
        else:
            seq = seq[:, -self.config.SEQ_LEN:]
        
        # モデル推論 - Phase III: 盤面も渡す
        temp = self.get_temperature(self.stats['total_games'])
        
        # 現在盤面を取得 (engine.boards: [B, 2, 19, 19] -> [B, 19, 19])
        current_boards = self.engine.boards[:, 0] - self.engine.boards[:, 1]  # 黒=1, 白=-1, 空=0
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model.get_move_probabilities(seq, current_boards, legal_mask, temperature=temp)
        # 手を選択
        moves = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 着手
        self.engine.play_batch(moves)
        self.game_move_counts += 1
        
        # 履歴に追加
        for i in range(self.config.BATCH_SIZE):
            self.game_histories[i].append(moves[i].clone())
        
        # 終局検出
        finished = self.engine.is_game_over()
        max_moves_reached = self.game_move_counts >= self.config.MAX_MOVES_PER_GAME
        should_end = finished | max_moves_reached
        
        # 終局したゲームの処理
        loss = 0.0
        num_finished = should_end.sum().item()
        
        if num_finished > 0:
            loss = self._process_finished_games(should_end)
        
        return num_finished, loss
    
    def _process_finished_games(self, finished_mask: torch.Tensor) -> float:
        """終局したゲームを処理（学習＆リセット）"""
        
        finished_indices = finished_mask.nonzero().squeeze(-1)
        if finished_indices.dim() == 0:
            finished_indices = finished_indices.unsqueeze(0)
        
        total_loss = 0.0
        num_learned = 0
        
        for idx in finished_indices:
            idx_item = idx.item()
            history = self.game_histories[idx_item]
            
            if len(history) > 1:
                # 学習
                loss = self._learn_from_game(history)
                total_loss += loss
                num_learned += 1
            
            # 履歴をクリア
            self.game_histories[idx_item] = []
        
        # ゲーム手数をリセット
        self.game_move_counts[finished_mask] = 0
        
        # 終局した盤面をリセット
        self.engine.reset_selected(finished_mask)
        
        # 統計更新
        self.stats['total_games'] += len(finished_indices)
        
        return total_loss / num_learned if num_learned > 0 else 0.0
    
    def _learn_from_game(self, history: List[torch.Tensor]) -> float:
        """1ゲームから学習 - Phase III 対応版（盤面再構成）"""
        
        if len(history) < 2:
            return 0.0
        
        # 履歴をテンソルに
        moves = torch.stack(history)  # (seq_len,)
        
        input_seq = moves[:-1].unsqueeze(0)  # (1, seq_len-1)
        target_seq = moves[1:].unsqueeze(0)  # (1, seq_len-1)
        
        # Phase III: 盤面を再構成
        with torch.no_grad():
            all_boards = self.engine.replay_history_to_boards_fast(moves[:-1])
            input_boards = all_boards.unsqueeze(0)  # (1, seq_len-1, 19, 19)
        
        # パディング
        if input_seq.shape[1] < self.config.SEQ_LEN:
            pad_len = self.config.SEQ_LEN - input_seq.shape[1]
            # シーケンスのパディング
            pad = torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long, device=self.device)
            input_seq = torch.cat([pad, input_seq], dim=1)
            target_pad = torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long, device=self.device)
            target_seq = torch.cat([target_pad, target_seq], dim=1)
            # 盤面のパディング（空盤面で埋める）
            board_pad = torch.zeros((1, pad_len, BOARD_SIZE, BOARD_SIZE), 
                                    dtype=torch.float32, device=self.device)
            input_boards = torch.cat([board_pad, input_boards], dim=1)
        
        # 学習
        self.model.train()
        self.optimizer.zero_grad()
        
        # Phase III: TesseraModel は (seq, board) を受け取る
        # 最後の盤面を使用（現在の盤面状態）
        current_board = input_boards[:, -1, :, :]  # (1, 19, 19)
        logits, = self.model(input_seq, current_board, return_value=False)
        logits = logits.contiguous()
        target_seq = target_seq.contiguous()
        
        # Phase III: TesseraModel は最後の一手のみ予測
        # target_seq は [1, seq_len] なので、最後の手だけを使用
        target_last = target_seq[:, -1]  # shape: [1]
        
        # logits は [1, VOCAB_SIZE] なので reshape 不要
        loss = self.criterion(logits, target_last)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        self.scheduler.step()
        
        loss_val = loss.item()
        
        # 統計更新
        self.stats['total_loss'] += loss_val
        self.stats['loss_count'] += 1
        self.stats['total_moves'] += len(history)
        
        if loss_val < self.stats['best_loss']:
            self.stats['best_loss'] = loss_val
        
        return loss_val

    def run_elo_evaluation(self, model_name: str) -> Optional[Dict]:
        """
        ELO評価を実行（Clean Room Protocol）
        
        Contract:
            この関数は学習用リソース（Engine, History）を一時的に解放し、
            Clean Room で ELO 評価を行った後、リソースを再構築する。
            
        Pipeline:
            1. Release: 学習用エンジンと履歴を解放
            2. Sanitize: VRAMをクリーンアップ
            3. Execute: ELO評価を実行
            4. Rebuild: 学習用リソースを再構築
        """
        import gc
        
        # === Step 1: Release Training Resources ===
        # 学習用エンジンを解放（VRAM の大部分を明け渡す）
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine
            self.engine = None
        
        # Game History をクリア（Tensor参照を切る）
        # これを忘れると「エンジンを消したのにVRAMが空かない」ゾンビ現象が起きる
        for i in range(len(self.game_histories)):
            self.game_histories[i] = []
        
        # Game Move Counts をリセット
        self.game_move_counts = torch.zeros(
            self.config.BATCH_SIZE, dtype=torch.long, device=self.device
        )
        
        # === Step 2: Sanitize VRAM ===
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # === Step 3: Execute ELO Evaluation ===
        self.model.eval()  # ELO評価前に eval モードに設定
        result = None
        try:
            # elo_evaluator.evaluate_and_update 内部で VRAMSanitizer が走る
            result = self.elo_evaluator.evaluate_and_update(self.model, model_name)
            
            if result:
                self.stats['current_elo'] = result['elo_after']
                if result['elo_after'] > self.stats['best_elo']:
                    self.stats['best_elo'] = result['elo_after']
                
                self.elo_logger.log_match(result)
                self.tile_elo_tracker.record_match(result, self.stats['total_games'])
        
        except Exception as e:
            print(f"❌ ELO evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # === Step 4: Rebuild Training Resources ===
        # 学習用エンジンの再構築（コスト: ~100ms、安定性のための必要経費）
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        self.engine = GPUGoEngine(batch_size=self.config.BATCH_SIZE, device=self.device)
        
        # Game Histories を再初期化
        self.game_histories = [[] for _ in range(self.config.BATCH_SIZE)]
        
        return result


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # GPU情報
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device = 'cuda'
    else:
        vram_gb = 0
        device = 'cpu'
    
    config = Config.auto_scale(vram_gb)
    
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    
    logger = Logger(config.LOG_DIR)
    shutdown_handler = GracefulShutdown()
    
    # ヘッダー
    logger.log("=" * 70)
    logger.log("🚀 Tessera Long Training v4.0 - Parallel + ELO")
    logger.log("=" * 70)
    logger.log(f"Device: {device}")
    
    if device == 'cuda':
        logger.log(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB)")
    
    logger.log(f"\n📊 Config:")
    logger.log(f"   Batch size (parallel games): {config.BATCH_SIZE}")
    logger.log(f"   Model: d={config.D_MODEL}, layers={config.N_LAYERS}")
    logger.log(f"   Target games: {config.NUM_GAMES}")
    logger.log(f"   Tile size: {config.TILE_SIZE}")
    logger.log(f"   ELO eval interval: {config.ELO_EVAL_INTERVAL}")
    
    # トレーナー初期化
    logger.log("\n📦 Initializing trainer...")
    trainer = ParallelTrainer(config, device)
    
    num_params = sum(p.numel() for p in trainer.model.parameters())
    logger.log(f"Model parameters: {num_params:,}")
    
    # チェックポイントから再開
    if args.resume:
        logger.log(f"\n📂 Resuming from {args.resume}...")
        trainer.stats = load_checkpoint(
            args.resume,
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            device
        )
        logger.log(f"   Resumed at game {trainer.stats.get('total_games', 0)}")
    
    # JSON ログに開始を記録
    logger.log_json("training_start", {
        "config": asdict(config),
        "vram_gb": vram_gb,
    })
    
    # 学習ループ
    logger.log("\n🎮 Starting parallel training loop...\n")
    
    start_time = time.time()
    last_log_games = trainer.stats['total_games']
    last_elo_games = trainer.stats['total_games']
    last_checkpoint_games = trainer.stats['total_games']
    last_tile_games = trainer.stats['total_games']
    
    losses_window = trainer.stats.get('losses_window', [])
    
    try:
        while trainer.stats['total_games'] < config.NUM_GAMES:
            
            if shutdown_handler.should_stop():
                logger.log("🛑 Graceful shutdown initiated...")
                break
            
            # 1ステップ実行
            finished, loss = trainer.step()
            
            if loss > 0:
                losses_window.append(loss)
                if len(losses_window) > 100:
                    losses_window.pop(0)
            
            current_games = trainer.stats['total_games']
            
            # ログ出力
            if current_games - last_log_games >= config.LOG_INTERVAL:
                elapsed = time.time() - start_time
                games_per_hour = current_games / (elapsed / 3600) if elapsed > 0 else 0
                eta_hours = (config.NUM_GAMES - current_games) / games_per_hour if games_per_hour > 0 else 0
                
                avg_loss = trainer.stats['total_loss'] / trainer.stats['loss_count'] if trainer.stats['loss_count'] > 0 else 0
                recent_loss = sum(losses_window) / len(losses_window) if losses_window else 0
                
                logger.log(
                    f"Game {current_games:6d}/{config.NUM_GAMES} | "
                    f"Loss: {recent_loss:.4f} (best: {trainer.stats['best_loss']:.4f}) | "
                    f"ELO: {trainer.stats['current_elo']:.0f} | "
                    f"Speed: {games_per_hour:.0f}/hr | "
                    f"ETA: {eta_hours:.1f}h"
                )
                
                last_log_games = current_games
            
            # ELO評価
            if current_games - last_elo_games >= config.ELO_EVAL_INTERVAL:
                model_name = f"game_{current_games:06d}"
                result = trainer.run_elo_evaluation(model_name)
                
                if result:
                    logger.log(
                        f"📊 ELO: {result['elo_before']:.0f} → {result['elo_after']:.0f} "
                        f"(vs {result['opponent_model']}, win_rate={result['win_rate']:.1%})"
                    )
                
                last_elo_games = current_games
            
            # タイル境界
            if current_games - last_tile_games >= config.TILE_SIZE:
                if trainer.tile_elo_tracker.tile_matches:
                    summary = trainer.tile_elo_tracker.close_tile(
                        current_games,
                        trainer.stats['current_elo']
                    )
                    trainer.elo_logger.log_tile_summary(asdict(summary))
                    logger.log(f"📦 Tile {summary.tile_id}: ELO {summary.elo_start:.0f}→{summary.elo_end:.0f}")
                
                last_tile_games = current_games
            
            # チェックポイント保存
            if current_games - last_checkpoint_games >= config.CHECKPOINT_INTERVAL:
                avg_loss = trainer.stats['total_loss'] / trainer.stats['loss_count'] if trainer.stats['loss_count'] > 0 else 0
                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"tessera_v4_game{current_games:06d}_elo{trainer.stats['current_elo']:.0f}.pth"
                )
                trainer.stats['losses_window'] = losses_window
                save_checkpoint(
                    trainer.model,
                    trainer.optimizer,
                    trainer.scheduler,
                    trainer.stats,
                    config,
                    ckpt_path
                )
                logger.log(f"💾 Checkpoint: {ckpt_path}")
                
                last_checkpoint_games = current_games
            
            # 定期的にデフラグ
            if current_games % 100 == 0:
                defragment_gpu_memory()
    
    except Exception as e:
        logger.log(f"\n❌ Error: {str(e)}")
        logger.log(traceback.format_exc())
    
    finally:
        # 最終保存
        elapsed = time.time() - start_time
        current_games = trainer.stats['total_games']
        avg_loss = trainer.stats['total_loss'] / trainer.stats['loss_count'] if trainer.stats['loss_count'] > 0 else 0
        
        final_ckpt = os.path.join(
            config.CHECKPOINT_DIR,
            f"tessera_v4_final_game{current_games}_elo{trainer.stats['current_elo']:.0f}.pth"
        )
        trainer.stats['losses_window'] = losses_window
        save_checkpoint(
            trainer.model,
            trainer.optimizer,
            trainer.scheduler,
            trainer.stats,
            config,
            final_ckpt
        )
        
        # サマリー
        logger.log("\n" + "=" * 70)
        logger.log("📊 Training Summary")
        logger.log("=" * 70)
        logger.log(f"   Games completed: {current_games}")
        logger.log(f"   Total moves: {trainer.stats['total_moves']:,}")
        logger.log(f"   Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
        logger.log(f"   Average loss: {avg_loss:.4f}")
        logger.log(f"   Best loss: {trainer.stats['best_loss']:.4f}")
        logger.log(f"   Final ELO: {trainer.stats['current_elo']:.0f}")
        logger.log(f"   Best ELO: {trainer.stats['best_elo']:.0f}")
        logger.log(f"   Final checkpoint: {final_ckpt}")
        
        logger.log_json("training_end", {
            "games_completed": current_games,
            "total_hours": elapsed / 3600,
            "avg_loss": avg_loss,
            "final_elo": trainer.stats['current_elo'],
        })
        
        logger.log("\n✅ Training complete!")


if __name__ == "__main__":
    main()
