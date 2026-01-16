"""
Tessera Phase III.2 Training Script
====================================
Tromp-Taylor ルール + Value Head 学習

Changes from v5:
- 捕獲なし（Tromp-Taylor）
- 終局時に地計算で勝敗決定
- Policy Loss + Value Loss

Version: 0.1.0
"""

import sys
sys.path.insert(0, '/app/src')

import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataclasses import dataclass
from typing import List, Tuple

from gpu_go_engine import GPUGoEngine, PASS_TOKEN, PAD_TOKEN, BOARD_SIZE
from tessera_model import TesseraModel
from chain_utils import compute_territory


@dataclass
class Config:
    # Training
    NUM_GAMES: int = 50000
    BATCH_SIZE: int = 32
    MAX_MOVES_PER_GAME: int = 150
    SEQ_LEN: int = 256
    
    # Learning
    LEARNING_RATE: float = 1e-4
    VALUE_LOSS_WEIGHT: float = 0.5  # α in Total Loss = Policy + α * Value
    GRADIENT_CLIP_NORM: float = 1.0
    
    # Logging
    LOG_INTERVAL: int = 100
    SAVE_INTERVAL: int = 5000
    
    # Temperature
    TEMP_START: float = 1.5
    TEMP_END: float = 0.5
    TEMP_DECAY_GAMES: int = 30000


class Phase3Trainer:
    """Phase III.2 Trainer with Value Head"""
    
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Model
        self.model = TesseraModel().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.policy_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        self.value_criterion = nn.MSELoss()
        
        # Engine
        self.engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
        
        # Game state tracking
        self.game_histories: List[List[torch.Tensor]] = [[] for _ in range(config.BATCH_SIZE)]
        self.game_move_counts = torch.zeros(config.BATCH_SIZE, dtype=torch.long, device=device)
        
        # Statistics
        self.stats = {
            'total_games': 0,
            'total_loss': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'loss_count': 0,
        }
    
    def get_temperature(self, total_games: int) -> float:
        progress = min(1.0, total_games / self.config.TEMP_DECAY_GAMES)
        return self.config.TEMP_START - (self.config.TEMP_START - self.config.TEMP_END) * progress
    
    def step(self) -> Tuple[int, float]:
        """1ステップ実行（全盤面に1手着手）"""
        B = self.config.BATCH_SIZE
        
        # モデルで手を選択
        self.model.eval()
        with torch.no_grad():
            # 現在の盤面を取得
            boards = self.engine.boards[:, 0] - self.engine.boards[:, 1]  # (B, 19, 19)
            
            # シーケンスを構築
            seq_list = []
            for i in range(B):
                if len(self.game_histories[i]) > 0:
                    seq = torch.stack(self.game_histories[i])[-self.config.SEQ_LEN:]
                else:
                    seq = torch.tensor([PAD_TOKEN], device=self.device)
                seq_list.append(seq)
            
            # パディング
            max_len = max(s.size(0) for s in seq_list)
            padded_seqs = torch.full((B, max_len), PAD_TOKEN, dtype=torch.long, device=self.device)
            for i, seq in enumerate(seq_list):
                padded_seqs[i, -seq.size(0):] = seq
            
            # 推論
            logits, = self.model(padded_seqs, boards, return_value=False)
            
            # 合法手マスク適用
            legal_mask = self.engine.get_legal_mask()  # (B, 362)
            logits_362 = logits[:, :362]
            logits_362[~legal_mask] = float('-inf')
            
            # 温度付きサンプリング
            temp = self.get_temperature(self.stats['total_games'])
            probs = torch.softmax(logits_362 / temp, dim=-1)
            moves = torch.multinomial(probs, 1).squeeze(1)
        
        # 着手実行
        self.engine.play_batch(moves)
        self.game_move_counts += 1
        
        # 履歴に追加
        for i in range(B):
            self.game_histories[i].append(moves[i].clone())
        
        # 終局検出
        finished_pass = self.engine.is_game_over()
        finished_max = self.game_move_counts >= self.config.MAX_MOVES_PER_GAME
        should_end = finished_pass | finished_max
        
        # 終局したゲームの処理
        loss = 0.0
        num_finished = should_end.sum().item()
        
        if num_finished > 0:
            loss = self._process_finished_games(should_end)
        
        return num_finished, loss
    
    def _process_finished_games(self, finished_mask: torch.Tensor) -> float:
        """終局したゲームから学習"""
        finished_indices = finished_mask.nonzero().squeeze(-1)
        
        if finished_indices.numel() == 0:
            return 0.0
        
        # 勝敗を計算（Tromp-Taylor）
        _, _, winners = self.engine.compute_score()  # (B,) +1=黒勝, -1=白勝
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_learned = 0
        
        for idx in finished_indices:
            idx_item = idx.item()
            history = self.game_histories[idx_item]
            winner = winners[idx_item].item()
            
            if len(history) >= 2:
                policy_loss, value_loss = self._learn_from_game(history, winner)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                num_learned += 1
            
            # リセット
            self.game_histories[idx_item] = []
        
        # ゲーム手数をリセット
        self.game_move_counts[finished_mask] = 0
        
        # 盤面をリセット
        self.engine.reset_selected(finished_mask)
        
        # 統計更新
        self.stats['total_games'] += len(finished_indices)
        
        if num_learned > 0:
            self.stats['policy_loss'] += total_policy_loss
            self.stats['value_loss'] += total_value_loss
            self.stats['total_loss'] += total_policy_loss + self.config.VALUE_LOSS_WEIGHT * total_value_loss
            self.stats['loss_count'] += num_learned
        
        return (total_policy_loss + self.config.VALUE_LOSS_WEIGHT * total_value_loss) / max(1, num_learned)
    
    def _learn_from_game(self, history: List[torch.Tensor], winner: float) -> Tuple[float, float]:
        """1ゲームから学習（Policy + Value）"""
        
        moves = torch.stack(history)  # (seq_len,)
        seq_len = len(moves)
        
        # 入力シーケンスとターゲット
        input_seq = moves[:-1].unsqueeze(0)  # (1, seq_len-1)
        target_move = moves[-1].unsqueeze(0)  # (1,)
        
        # 盤面を再構成
        with torch.no_grad():
            all_boards = self.engine.replay_history_to_boards_fast(moves[:-1])
            current_board = all_boards[-1:] if len(all_boards) > 0 else torch.zeros(1, BOARD_SIZE, BOARD_SIZE, device=self.device)
        
        # パディング
        if input_seq.shape[1] < self.config.SEQ_LEN:
            pad_len = self.config.SEQ_LEN - input_seq.shape[1]
            pad = torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long, device=self.device)
            input_seq = torch.cat([pad, input_seq], dim=1)
        
        # 学習
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward（Policy + Value）
        policy_logits, value = self.model(input_seq, current_board, return_value=True)
        
        # Policy Loss
        policy_loss = self.policy_criterion(policy_logits, target_move)
        
        # Value Loss
        # 手番に応じて勝敗を調整（黒番で始まり、最後の手番の視点）
        # 履歴の長さが偶数なら白番、奇数なら黒番
        perspective = 1.0 if (seq_len - 1) % 2 == 0 else -1.0
        target_value = torch.tensor([[winner * perspective]], dtype=torch.float32, device=self.device)
        value_loss = self.value_criterion(value, target_value)
        
        # Total Loss
        total_loss = policy_loss + self.config.VALUE_LOSS_WEIGHT * value_loss
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return 0.0, 0.0
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        
        return policy_loss.item(), value_loss.item()


def main():
    print("=" * 60)
    print("Phase III.2 Training (Tromp-Taylor + Value Head)")
    print("=" * 60)
    
    config = Config()
    config.NUM_GAMES = 10000  # テスト用
    config.LOG_INTERVAL = 100
    config.SAVE_INTERVAL = 2000
    
    trainer = Phase3Trainer(config, device='cuda')
    
    # Phase II の重みをロード（オプション）
    checkpoint_path = '/app/checkpoints/tessera_v4_game802591_elo1490.pth'
    try:
        info = trainer.model.load_phase2_weights(checkpoint_path)
        print(f"Phase II weights loaded: {info['loaded_keys']} keys")
    except Exception as e:
        print(f"Starting fresh: {e}")
    
    print(f"Target: {config.NUM_GAMES} games")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max moves: {config.MAX_MOVES_PER_GAME}")
    print(f"Value loss weight: {config.VALUE_LOSS_WEIGHT}")
    print("Starting...\n")
    
    start = time.time()
    last_logged = 0
    
    while trainer.stats['total_games'] < config.NUM_GAMES:
        finished, loss = trainer.step()
        games = trainer.stats['total_games']
        
        if games >= last_logged + config.LOG_INTERVAL:
            elapsed = time.time() - start
            gps = games / elapsed
            
            avg_loss = trainer.stats['total_loss'] / max(1, trainer.stats['loss_count'])
            avg_policy = trainer.stats['policy_loss'] / max(1, trainer.stats['loss_count'])
            avg_value = trainer.stats['value_loss'] / max(1, trainer.stats['loss_count'])
            
            print(f"Game {games:5d} | Loss: {avg_loss:.4f} (P:{avg_policy:.4f} V:{avg_value:.4f}) | Speed: {gps:.1f} g/s")
            last_logged = games
        
        # チェックポイント保存
        if games > 0 and games % config.SAVE_INTERVAL == 0:
            save_path = f'/app/checkpoints/tessera_phase3.2_game{games}.pth'
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")
    
    # 最終保存
    elapsed = time.time() - start
    avg_loss = trainer.stats['total_loss'] / max(1, trainer.stats['loss_count'])
    save_path = f'/app/checkpoints/tessera_phase3.2_final_loss{avg_loss:.2f}.pth'
    torch.save(trainer.model.state_dict(), save_path)
    
    print()
    print("=" * 60)
    print(f"Training complete: {trainer.stats['total_games']} games in {elapsed/60:.1f} min")
    print(f"Final Loss: {avg_loss:.4f}")
    print(f"Speed: {trainer.stats['total_games']/elapsed:.1f} g/s")
    print(f"Saved: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
