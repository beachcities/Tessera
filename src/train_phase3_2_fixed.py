"""
Tessera Phase III.2 Training Script (Fixed + Vectorized)
=========================================================
Tromp-Taylor ルール + Value Head 学習 + Vectorized History

Changes from Fixed v0.2.0:
1. VectorizedGameHistory導入による高速化と堅牢化 (DEC-011)
2. 履歴管理と手数管理のアトミック化

Version: 0.3.0
"""

import sys
import os
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Path setup
sys.path.insert(0, '/app/src')

import torch
import torch.nn as nn
import torch.optim as optim

# Project imports
from eval_quick import quick_eval
from utils import get_turn_sequence
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, PAD_TOKEN, BOARD_SIZE
from tessera_model import TesseraModel
from chain_utils import compute_territory


class VectorizedGameHistory:
    """
    Manages game move histories and move counts using fixed-size preallocated tensors.
    
    Complies with Tessera Implementation Principles v1.0:
    - [I. Contract] Statefulness Boundary: Manages history AND move counts atomically.
    - [II. Tensor Integrity] Shape & Semantic Documentation.
    - [III. Data Representation] Right-aligned padding (Newest moves at right).
    - [VII. Performance] Preallocation & Vectorization (No Python loops).
    """
    def __init__(
        self, 
        batch_size: int, 
        seq_len: int, 
        padding_idx: int = PAD_TOKEN,
        device: str = 'cuda'
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.padding_idx = padding_idx
        self.device = torch.device(device)

        # [VII. Performance] Preallocation (History)
        # Shape: [B, SEQ_LEN]
        self.history_buffer = torch.full(
            (batch_size, seq_len), 
            padding_idx, 
            dtype=torch.long, 
            device=self.device
        )

        # [VII. Performance] Preallocation (Move Counts)
        # Shape: [B]
        self.move_counts = torch.zeros(
            (batch_size,), 
            dtype=torch.long, 
            device=self.device
        )

    def step(self, new_moves: torch.Tensor) -> torch.Tensor:
        """Updates history by shifting left and appending new moves."""
        if new_moves.dim() == 1:
            new_moves = new_moves.unsqueeze(1)
            
        # [II. Tensor Integrity] .clone() for safety
        self.history_buffer[:, :-1] = self.history_buffer[:, 1:].clone()
        self.history_buffer[:, -1] = new_moves.squeeze(1)
        
        self.move_counts += 1
        return self.history_buffer

    def reset_games(self, reset_mask: torch.Tensor):
        """Resets histories and move counts for ended games (Atomic)."""
        if reset_mask.any():
            self.history_buffer[reset_mask] = self.padding_idx
            self.move_counts[reset_mask] = 0

    def get_counts(self) -> torch.Tensor:
        return self.move_counts

    def get_history(self) -> torch.Tensor:
        return self.history_buffer

    def get_valid_moves(self, batch_idx: int) -> torch.Tensor:
        """
        Retrieves the valid moves for a specific game (removing padding).
        Used for learning from finished games.
        """
        count = self.move_counts[batch_idx].item()
        if count == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        valid_len = min(count, self.seq_len)
        return self.history_buffer[batch_idx, -valid_len:]


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
        
        self.history_manager = VectorizedGameHistory(
            batch_size=config.BATCH_SIZE,
            seq_len=config.SEQ_LEN,
            padding_idx=PAD_TOKEN,
            device=device
        )
        
        self.stats = {
            'total_games': 0, 'total_loss': 0.0, 'policy_loss': 0.0,
            'value_loss': 0.0, 'loss_count': 0, 'pass_count': 0, 'normal_count': 0,
        }

    def get_temperature(self, total_games: int) -> float:
        progress = min(1.0, total_games / self.config.TEMP_DECAY_GAMES)
        return self.config.TEMP_START - (self.config.TEMP_START - self.config.TEMP_END) * progress

    def step(self) -> Tuple[int, float]:
        self.model.eval()
        
        with torch.no_grad():
            raw_boards = self.engine.boards[:, 0] - self.engine.boards[:, 1]
            turn_perspective = torch.where(self.engine.turn == 0, 1.0, -1.0).view(-1, 1, 1)
            boards = raw_boards * turn_perspective
            
            padded_seqs = self.history_manager.get_history()
            move_counts = self.history_manager.get_counts()
            
            turn_seq = get_turn_sequence(move_counts, self.engine.turn, padded_seqs.shape[1], self.device)
            
            logits, = self.model(padded_seqs, boards, turn_seq=turn_seq, return_value=False)
            
            legal_mask = self.engine.get_legal_mask()
            logits_362 = logits[:, :362]
            pass_idx = 361
            
            under_min_mask = move_counts < self.config.MIN_MOVES_BEFORE_PASS
            has_moves = legal_mask[:, :361].any(dim=1)
            mask_pass_condition = under_min_mask & has_moves
            legal_mask[mask_pass_condition, pass_idx] = False

            logits_362[~legal_mask] = float('-inf')
            
            temp = self.get_temperature(self.stats['total_games'])
            probs = torch.softmax(logits_362 / temp, dim=-1)
            moves = torch.multinomial(probs, 1).squeeze(1)
            
            is_pass = (moves == pass_idx)
            self.stats['pass_count'] += is_pass.sum().item()
            self.stats['normal_count'] += (~is_pass).sum().item()

        self.engine.play_batch(moves)
        self.history_manager.step(moves)
        
        finished_pass = self.engine.is_game_over()
        finished_max = self.history_manager.get_counts() >= self.config.MAX_MOVES_PER_GAME
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
            
            history_tensor = self.history_manager.get_valid_moves(idx_item)
            winner = winners[idx_item].item()
            
            if history_tensor.size(0) >= 4:
                policy_loss, value_loss = self._learn_from_game_full(history_tensor, winner)
                total_policy_loss += policy_loss
                total_value_loss += value_loss
                num_learned += 1
            
        self.history_manager.reset_games(finished_mask)
        self.engine.reset_selected(finished_mask)
        self.stats['total_games'] += len(finished_indices)
        
        if num_learned > 0:
            self.stats['policy_loss'] += total_policy_loss
            self.stats['value_loss'] += total_value_loss
            self.stats['total_loss'] += total_policy_loss + self.config.VALUE_LOSS_WEIGHT * total_value_loss
            self.stats['loss_count'] += num_learned
            
        return (total_policy_loss + self.config.VALUE_LOSS_WEIGHT * total_value_loss) / max(1, num_learned)

    def _learn_from_game_full(self, moves: torch.Tensor, winner: float) -> Tuple[float, float]:
        seq_len = moves.size(0)
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
            else:
                input_seq = input_seq[:, -self.config.SEQ_LEN:]
                
            with torch.no_grad():
                # [FIX] Tensor directly, not wrapped in list
                all_boards = self.engine.replay_history_to_boards_fast(input_moves)
                
                if len(all_boards) > 0:
                    current_board = all_boards[-1:]
                else:
                    current_board = torch.zeros(1, BOARD_SIZE, BOARD_SIZE, device=self.device)
            
            perspective = 1.0 if idx % 2 == 0 else -1.0
            current_board = current_board * perspective
            current_player = idx % 2
            
            history_lengths = torch.tensor([len(input_moves)], device=self.device)
            current_turns = torch.tensor([current_player], device=self.device)
            
            turn_seq = get_turn_sequence(history_lengths, current_turns, input_seq.shape[1], self.device)
            
            policy_logits, value = self.model(input_seq, current_board, turn_seq=turn_seq, return_value=True)
            
            raw_policy_loss = self.policy_criterion(policy_logits, target_move.unsqueeze(0))
            policy_loss = raw_policy_loss * weight
            
            target_value = torch.tensor([[winner * perspective]], dtype=torch.float32, device=self.device)
            value_loss = self.value_criterion(value, target_value)
            
            total_policy_loss += raw_policy_loss.item() * weight
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
    print("Phase III.2 Training (FIXED + Vectorized v0.3.0)")
    print("=" * 60)
    
    config = Config()
    config.NUM_GAMES = 50000
    config.LOG_INTERVAL = 256
    config.SAVE_INTERVAL = 5000
    
    trainer = Phase3Trainer(config, device='cuda')
    
    checkpoint_path = '/app/checkpoints/tessera_phase3.2_fixed_latest.pth'
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path)
            trainer.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded: {checkpoint_path}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Starting fresh.")
    else:
        print("No checkpoint found. Starting fresh.")

    print(f"Target: {config.NUM_GAMES} games")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("Starting...\n")
    
    start = time.time()
    last_logged = 0
    last_saved = 0
    
    while trainer.stats['total_games'] < config.NUM_GAMES:
        finished, loss = trainer.step()
        
        games = trainer.stats['total_games']
        
        if games >= last_logged + config.LOG_INTERVAL:
            elapsed = time.time() - start
            gps = games / elapsed if elapsed > 0 else 0
            
            avg_loss = trainer.stats['total_loss'] / max(1, trainer.stats['loss_count'])
            avg_policy = trainer.stats['policy_loss'] / max(1, trainer.stats['loss_count'])
            avg_value = trainer.stats['value_loss'] / max(1, trainer.stats['loss_count'])
            
            total_moves = trainer.stats['pass_count'] + trainer.stats['normal_count']
            pass_rate = trainer.stats['pass_count'] / max(1, total_moves) * 100
            
            print(f"Game {games:5d} | Loss: {avg_loss:.4f} (P:{avg_policy:.4f} V:{avg_value:.4f}) | Pass: {pass_rate:.1f}% | Speed: {gps:.1f} g/s")
            
            win_rate = quick_eval(trainer.model, device="cuda", num_games=64, verbose=False)
            print(f"          -> Win Rate: {win_rate*100:.1f}%")
            
            last_logged = games
            
        if games > 0 and games % config.SAVE_INTERVAL == 0 and games > last_saved:
            save_path = f'/app/checkpoints/tessera_phase3.2_fixed_game{games}.pth'
            torch.save(trainer.model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")
            last_saved = games

    elapsed = time.time() - start
    avg_loss = trainer.stats['total_loss'] / max(1, trainer.stats['loss_count'])
    save_path = f'/app/checkpoints/tessera_phase3.2_fixed_final_loss{avg_loss:.2f}.pth'
    torch.save(trainer.model.state_dict(), save_path)
    
    print()
    print("=" * 60)
    print(f"Training complete: {trainer.stats['total_games']} games in {elapsed/60:.1f} min")
    print(f"Final Loss: {avg_loss:.4f}")
    print(f"Saved: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
