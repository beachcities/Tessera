"""
Tessera Phase III Long Training Script v5.0.0
=============================================
Phase II の long_training_v4.py をベースに
TesseraModel（Mamba + Diffusion + RayCast + Fusion）を統合

段階的解凍（Progressive Unfreezing）:
- Phase 3.1 (0-2000 games): Mamba フリーズ、新規層のみ学習
- Phase 3.2 (2000-12000 games): Mamba 解凍（LR 1/10）
- Phase 3.3 (12000+ games): 全体統合学習

Version: 5.0.0
"""

import os
import sys
import signal
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_go_engine import GPUGoEngine, PASS_TOKEN, VOCAB_SIZE, PAD_TOKEN, BOARD_SIZE
from tessera_model import TesseraModel
from diffusion import BOARD_SIZE as DIFF_BOARD_SIZE

# ============================================================
# Phase Configuration
# ============================================================

@dataclass
class PhaseConfig:
    PHASE_3_1 = "3.1"
    PHASE_3_2 = "3.2"  
    PHASE_3_3 = "3.3"
    PHASE_3_1_END = 2000
    PHASE_3_2_END = 12000

# ============================================================
# Training Configuration
# ============================================================

@dataclass
class Config:
    NUM_GAMES: int = 10000
    MAX_MOVES_PER_GAME: int = 200
    SEQ_LEN: int = 64
    BATCH_SIZE: int = 32
    
    # Model
    D_MODEL: int = 256
    N_LAYERS: int = 4
    VOCAB_SIZE: int = 364
    
    # Learning rates
    LR_NEW_LAYERS: float = 1e-4
    LR_MAMBA_UNFROZEN: float = 1e-5
    LR_UNIFIED: float = 5e-5
    GRADIENT_CLIP: float = 0.5
    
    # Logging
    LOG_INTERVAL: int = 100
    CHECKPOINT_INTERVAL: int = 1000
    
    # Temperature
    TEMP_START: float = 1.5
    TEMP_END: float = 0.8
    TEMP_DECAY_GAMES: int = 5000
    
    CHECKPOINT_DIR: str = "/app/checkpoints"
    LOG_DIR: str = "/app/logs"

# ============================================================
# Graceful Shutdown
# ============================================================

class GracefulShutdown:
    def __init__(self):
        self._stop = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)
    
    def _handler(self, signum, frame):
        print("\n[SHUTDOWN] Signal received, finishing current game...")
        self._stop = True
    
    def should_stop(self) -> bool:
        return self._stop

# ============================================================
# Phase III Trainer
# ============================================================

class TesseraTrainerPhase3:
    def __init__(self, config: Config, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.phase = PhaseConfig.PHASE_3_1
        self.total_games = 0
        self.total_steps = 0
        
        # Paths
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.log_dir = Path(config.LOG_DIR)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Model
        print("[INIT] Creating TesseraModel...")
        self.model = TesseraModel(
            vocab_size=config.VOCAB_SIZE,
            d_model=config.D_MODEL,
            n_layers=config.N_LAYERS,
        ).to(device)
        
        # Load Phase II weights
        print("[INIT] Loading Phase II weights...")
        phase2_path = self.checkpoint_dir / "tessera_v4_game802591_elo1490.pth"
        if phase2_path.exists():
            info = self.model.load_phase2_weights(str(phase2_path))
            print(f"[INIT] Loaded {info['loaded_keys']} keys from Phase II")
        else:
            print("[WARN] Phase II checkpoint not found, starting fresh")
        
        # Engine
        print("[INIT] Creating GPUGoEngine...")
        self.engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
        
        # Game state
        self.game_histories: List[List[torch.Tensor]] = [[] for _ in range(config.BATCH_SIZE)]
        self.game_move_counts = torch.zeros(config.BATCH_SIZE, dtype=torch.long, device=device)
        
        # Stats
        self.stats = {'total_games': 0, 'total_loss': 0.0, 'loss_count': 0}
        
        # Setup initial phase
        self._setup_phase(PhaseConfig.PHASE_3_1)
        print(f"[INIT] Phase {self.phase} started, Mamba frozen: {self._is_mamba_frozen()}")
        
        # Game state
        self.game_histories: List[List[torch.Tensor]] = [[] for _ in range(config.BATCH_SIZE)]
        self.game_move_counts = torch.zeros(config.BATCH_SIZE, dtype=torch.long, device=device)
        
        # Stats
        self.stats = {'total_games': 0, 'total_loss': 0.0, 'loss_count': 0}
        
        # Setup initial phase
        self._setup_phase(PhaseConfig.PHASE_3_1)
        print(f"[INIT] Phase {self.phase} started, Mamba frozen: {self._is_mamba_frozen()}")

    def _is_mamba_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.mamba.parameters())

    def _set_mamba_grad(self, requires_grad: bool):
        for p in self.model.mamba.parameters():
            p.requires_grad = requires_grad

    def _new_layer_params(self):
        return [p for n, p in self.model.named_parameters() if not n.startswith('mamba.')]

    def _setup_phase(self, phase: str):
        self.phase = phase
        if phase == PhaseConfig.PHASE_3_1:
            self._set_mamba_grad(False)
            self.optimizer = optim.AdamW(self._new_layer_params(), lr=self.config.LR_NEW_LAYERS)
        elif phase == PhaseConfig.PHASE_3_2:
            self._set_mamba_grad(True)
            self.optimizer = optim.AdamW([
                {'params': self.model.mamba.parameters(), 'lr': self.config.LR_MAMBA_UNFROZEN},
                {'params': self._new_layer_params(), 'lr': self.config.LR_NEW_LAYERS}
            ])
        else:
            self._set_mamba_grad(True)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.LR_UNIFIED)

    def _update_phase(self):
        if self.phase == PhaseConfig.PHASE_3_1 and self.total_games >= PhaseConfig.PHASE_3_1_END:
            self._save_checkpoint("p3.1_final")
            self._setup_phase(PhaseConfig.PHASE_3_2)
            print(f"[PHASE] Transitioned to Phase 3.2, Mamba unfrozen (LR 1/10)")
        elif self.phase == PhaseConfig.PHASE_3_2 and self.total_games >= PhaseConfig.PHASE_3_2_END:
            self._save_checkpoint("p3.2_final")
            self._setup_phase(PhaseConfig.PHASE_3_3)
            print(f"[PHASE] Transitioned to Phase 3.3, unified learning")

    def _save_checkpoint(self, tag: str):
        path = self.checkpoint_dir / f"tessera_v5_{tag}_game{self.total_games}.pth"
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'phase': self.phase,
            'total_games': self.total_games,
            'stats': self.stats,
        }, path)
        print(f"[SAVE] {path.name}")

    def get_temperature(self, games: int) -> float:
        if games >= self.config.TEMP_DECAY_GAMES:
            return self.config.TEMP_END
        ratio = games / self.config.TEMP_DECAY_GAMES
        return self.config.TEMP_START + (self.config.TEMP_END - self.config.TEMP_START) * ratio
