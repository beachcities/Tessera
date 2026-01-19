"""
Tessera Phase III.2 Training Script (Fixed)
============================================
Tromp-Taylor ルール + Value Head 学習

Fixes:
1. ゲーム全体から学習（最後の1手だけでなく）
2. 序盤・中盤のパス禁止
3. パス手の学習重み低減
4. Value Headを全手番で学習

Version: 0.2.0
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
    NUM_GAMES: int = 50000
    BATCH_SIZE: int = 32
    MAX_MOVES_PER_GAME: int = 150
    SEQ_LEN: int = 256
    LEARNING_RATE: float = 1e-4
    VALUE_LOSS_WEIGHT: float = 0.5
    GRADIENT_CLIP_NORM: float = 1.0
    MIN_MOVES_BEFORE_PASS: int = 50
    PASS_PENALTY_WEIGHT: float = 0.1
    LEARN_SAMPLES_PER_GAME: int = 8
    LOG_INTERVAL: int = 100
    SAVE_INTERVAL: int = 5000
    TEMP_START: float = 1.5
    TEMP_END: float = 0.5
    TEMP_DECAY_GAMES: int = 30000


class Phase3Trainer:
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.model = TesseraModel().to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.policy_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, reduction='none')
        self.value_criterion = nn.MSELoss()
        self.engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
        self.game_histories: List[List[torch.Tensor]] = [[] for _ in range(config.BATCH_SIZE)]
        self.game_move_counts = torch.zeros(config.BATCH_SIZE, dtype=torch.long, device=device)
        self.stats = {
            'total_games': 0, 'total_loss': 0.0, 'policy_loss': 0.0,
            'value_loss': 0.0, 'loss_count': 0, 'pass_count': 0, 'normal_count': 0,
        }

    def get_temperature(self, total_games: int) -> float:
        progress = min(1.0, total_games / self.config.TEMP_DECAY_GAMES)
        return self.config.TEMP_START - (self.config.TEMP_START - self.config.TEMP_END) * progress

    def step(self) -> Tuple[int, float]:
        B = self.config.BATCH_SIZE
        self.model.eval()
        with torch.no_grad():
            boards = self.engine.boards[:, 0] - self.engine.boards[:, 1]
            seq_list = []
            for i in range(B):
                if len(self.game_histories[i]) > 0:
                    seq = torch.stack(self.game_histories[i])[-self.config.SEQ_LEN:]
                else:
                    seq = torch.tensor([PAD_TOKEN], device=self.device)
                seq_list.append(seq)
            max_len = max(s.size(0) for s in seq_list)
            padded_seqs = torch.full((B, max_len), PAD_TOKEN, dtype=torch.long, device=self.device)
            for i, seq in enumerate(seq_list):
                padded_seqs[i, -seq.size(0):] = seq
            logits, = self.model(padded_seqs, boards, return_value=False)
            legal_mask = self.engine.get_legal_mask()
            logits_362 = logits[:, :362]
            pass_idx = 361
            for i in range(B):
                move_count = self.game_move_counts[i].item()
                if move_count < self.config.MIN_MOVES_BEFORE_PASS:
                    non_pass_legal = legal_mask[i, :361].any()
                    if non_pass_legal:
                        legal_mask[i, pass_idx] = False
            logits_362[~legal_mask] = float('-inf')
            temp = self.get_temperature(self.stats['total_games'])
            probs = torch.softmax(logits_362 / temp, dim=-1)
            moves = torch.multinomial(probs, 1).squeeze(1)
            is_pass = (moves == pass_idx)
            self.stats['pass_count'] += is_pass.sum().item()
            self.stats['normal_count'] += (~is_pass).sum().item()
        self.engine.play_batch(moves)
        self.game_move_counts += 1
        for i in range(B):
            self.game_histories[i].append(moves[i].clone())
        finished_pass = self.engine.is_game_over()
        finished_max = self.game_move_counts >= self.config.MAX_MOVES_PER_GAME
        should_end = finished_pass | finished_max
        loss = 0.0
        num_finished = should_end.sum().item()
        if num_finished > 0:
            loss = self._process_finished_games(should_end)
        return num_finished, loss

    def _process_finished_games(self, finished_mask: torch.Tensor) -> float:
        finished_indices = finished_mask.nonzero().squeeze(-1)
        if finished_indices.numel() == 0:
            return 0.0
        _, _, winners = self.engine.compute_score()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_learned = 0
        for idx in finished_indices:
            idx_item = idx.item()
            history = self.game_histories[idx_item]
            winner = winners[idx_item].item()
            if len(history) >= 4:
                policy_loss, value_loss = self._learn_from_game_full(history, winner)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                num_learned += 1
            self.game_histories[idx_item] = []
        self.game_move_counts[finished_mask] = 0
        self.engine.reset_selected(finished_mask)
        self.stats['total_games'] += len(finished_indices)
        if num_learned > 0:
            self.stats['policy_loss'] += total_policy_loss
            self.stats['value_loss'] += total_value_loss
            self.stats['total_loss'] += total_policy_loss + self.config.VALUE_LOSS_WEIGHT * total_value_loss
            self.stats['loss_count'] += num_learned
        return (total_policy_loss + self.config.VALUE_LOSS_WEIGHT * total_value_loss) / max(1, num_learned)

    def _learn_from_game_full(self, history: List[torch.Tensor], winner: float) -> Tuple[float, float]:
        moves = torch.stack(history)
        seq_len = len(moves)
        valid_end = max(2, seq_len - 2)
        num_samples = min(self.config.LEARN_SAMPLES_PER_GAME, valid_end - 1)
        if num_samples <= 0:
            return 0.0, 0.0
        sample_indices = torch.linspace(1, valid_end - 1, num_samples).long()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_weight = 0.0
        self.model.train()
        self.optimizer.zero_grad()
        for sample_idx in sample_indices:
            idx = sample_idx.item()
            input_moves = moves[:idx]
            target_move = moves[idx]
            is_pass = (target_move.item() == 361)
            weight = self.config.PASS_PENALTY_WEIGHT if is_pass else 1.0
            input_seq = input_moves.unsqueeze(0)
            if input_seq.shape[1] < self.config.SEQ_LEN:
                pad_len = self.config.SEQ_LEN - input_seq.shape[1]
                pad = torch.full((1, pad_len), PAD_TOKEN, dtype=torch.long, device=self.device)
                input_seq = torch.cat([pad, input_seq], dim=1)
            with torch.no_grad():
                all_boards = self.engine.replay_history_to_boards_fast(input_moves)
                if len(all_boards) > 0:
                    current_board = all_boards[-1:]
                else:
                    current_board = torch.zeros(1, BOARD_SIZE, BOARD_SIZE, device=self.device)
            policy_logits, value = self.model(input_seq, current_board, return_value=True)
            policy_loss = self.policy_criterion(policy_logits, target_move.unsqueeze(0))
            policy_loss = policy_loss * weight
            perspective = 1.0 if idx % 2 == 0 else -1.0
            target_value = torch.tensor([[winner * perspective]], dtype=torch.float32, device=self.device)
            value_loss = self.value_criterion(value, target_value)
            total_policy_loss += policy_loss.item() * weight
            total_value_loss += value_loss.item()
            total_weight += weight
            loss = policy_loss + self.config.VALUE_LOSS_WEIGHT * value_loss
            if not (torch.isnan(loss) or torch.isinf(loss)):
                loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRADIENT_CLIP_NORM)
        self.optimizer.step()
        return total_policy_loss / max(1, total_weight), total_value_loss / max(1, num_samples)


def main():
    print("=" * 60)
    print("Phase III.2 Training (FIXED - Full Game Learning)")
    print("=" * 60)
    config = Config()
    config.NUM_GAMES = 10000
    config.LOG_INTERVAL = 100
    config.SAVE_INTERVAL = 2000
    config.MIN_MOVES_BEFORE_PASS = 50
    config.PASS_PENALTY_WEIGHT = 0.1
    config.LEARN_SAMPLES_PER_GAME = 8
    trainer = Phase3Trainer(config, device='cuda')
    checkpoint_path = '/app/checkpoints/tessera_phase3.2_final_loss3.58.pth'
    try:
        trainer.model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded: {checkpoint_path}")
    except Exception as e:
        print(f"Starting fresh: {e}")
    print(f"Target: {config.NUM_GAMES} games")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max moves: {config.MAX_MOVES_PER_GAME}")
    print(f"Min moves before pass: {config.MIN_MOVES_BEFORE_PASS}")
    print(f"Pass penalty weight: {config.PASS_PENALTY_WEIGHT}")
    print(f"Samples per game: {config.LEARN_SAMPLES_PER_GAME}")
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
            total_moves = trainer.stats['pass_count'] + trainer.stats['normal_count']
            pass_rate = trainer.stats['pass_count'] / max(1, total_moves) * 100
            print(f"Game {games:5d} | Loss: {avg_loss:.4f} (P:{avg_policy:.4f} V:{avg_value:.4f}) | Pass: {pass_rate:.1f}% | Speed: {gps:.1f} g/s")
            last_logged = games
        if games > 0 and games % config.SAVE_INTERVAL == 0:
            save_path = f'/app/checkpoints/tessera_phase3.2_fixed_game{games}.pth'
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")
    elapsed = time.time() - start
    avg_loss = trainer.stats['total_loss'] / max(1, trainer.stats['loss_count'])
    save_path = f'/app/checkpoints/tessera_phase3.2_fixed_final_loss{avg_loss:.2f}.pth'
    torch.save(trainer.model.state_dict(), save_path)
    print()
    print("=" * 60)
    print(f"Training complete: {trainer.stats['total_games']} games in {elapsed/60:.1f} min")
    print(f"Final Loss: {avg_loss:.4f}")
    print(f"Final Pass Rate: {trainer.stats['pass_count'] / max(1, trainer.stats['pass_count'] + trainer.stats['normal_count']) * 100:.1f}%")
    print(f"Speed: {trainer.stats['total_games']/elapsed:.1f} g/s")
    print(f"Saved: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
