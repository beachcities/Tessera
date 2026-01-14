"""Tessera Phase III Trainer - Progressive Unfreezing"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime
from tessera_model import TesseraModel
from diffusion import BOARD_SIZE

class PhaseConfig:
    PHASE_3_1 = "3.1"
    PHASE_3_2 = "3.2"
    PHASE_3_3 = "3.3"
    PHASE_3_1_END = 2000
    PHASE_3_2_END = 12000

class TesseraTrainerPhase3:
    def __init__(self, model, device='cuda', lr_new=1e-4, lr_mamba=1e-5, lr_unified=5e-5):
        self.model = model.to(device)
        self.device = device
        self.lr_new = lr_new
        self.lr_mamba = lr_mamba
        self.lr_unified = lr_unified
        self.checkpoint_dir = Path('/app/checkpoints')
        self.log_dir = Path('/app/logs')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.phase = PhaseConfig.PHASE_3_1
        self.total_games = 0
        self._setup_phase(self.phase)

    def _log(self, msg):
        ts = datetime.now().strftime('%H:%M:%S')
        print(f"[{ts}] {msg}")

    def _set_mamba_grad(self, requires_grad):
        for p in self.model.mamba.parameters():
            p.requires_grad = requires_grad
        self._log(f"Mamba: {'Unfrozen' if requires_grad else 'Frozen'}")

    def _new_params(self):
        return [p for n, p in self.model.named_parameters() if not n.startswith('mamba.')]

    def _setup_phase(self, phase):
        self.phase = phase
        self._log(f"Phase {phase} started")
        if phase == PhaseConfig.PHASE_3_1:
            self._set_mamba_grad(False)
            self.opt = optim.AdamW(self._new_params(), lr=self.lr_new)
        elif phase == PhaseConfig.PHASE_3_2:
            self._set_mamba_grad(True)
            self.opt = optim.AdamW([
                {'params': self.model.mamba.parameters(), 'lr': self.lr_mamba},
                {'params': self._new_params(), 'lr': self.lr_new}])
        else:
            self._set_mamba_grad(True)
            self.opt = optim.AdamW(self.model.parameters(), lr=self.lr_unified)

    def _verify_mamba_frozen(self):
        for n, p in self.model.mamba.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                return False
        return True

    def _update_phase(self):
        if self.phase == PhaseConfig.PHASE_3_1 and self.total_games >= PhaseConfig.PHASE_3_1_END:
            self._save_checkpoint("p3.1_final")
            self._setup_phase(PhaseConfig.PHASE_3_2)
        elif self.phase == PhaseConfig.PHASE_3_2 and self.total_games >= PhaseConfig.PHASE_3_2_END:
            self._save_checkpoint("p3.2_final")
            self._setup_phase(PhaseConfig.PHASE_3_3)

    def _save_checkpoint(self, tag):
        path = self.checkpoint_dir / f"tessera_v5_{tag}_game{self.total_games}.pth"
        torch.save({'model': self.model.state_dict(), 'opt': self.opt.state_dict(),
                    'phase': self.phase, 'games': self.total_games}, path)
        self._log(f"Saved: {path.name}")

    def train_step(self, moves, board, policy):
        self.model.train()
        self.opt.zero_grad()
        pred, = self.model(moves, board, return_value=False)
        loss = nn.functional.cross_entropy(pred, policy.argmax(dim=1))
        loss.backward()
        if self.phase == PhaseConfig.PHASE_3_1 and not self._verify_mamba_frozen():
            raise RuntimeError("Mamba gradient leak!")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def train_game(self, data):
        loss = self.train_step(data['moves'].to(self.device), data['board'].to(self.device), data['policy'].to(self.device))
        self.total_games += 1
        self._update_phase()
        return {'loss': loss, 'phase': self.phase, 'games': self.total_games}

    def status(self):
        return {'phase': self.phase, 'games': self.total_games, 'mamba_frozen': not any(p.requires_grad for p in self.model.mamba.parameters())}

    def _verify_mamba_frozen(self):
        for n, p in self.model.mamba.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                return False
        return True

    def _update_phase(self):
        if self.phase == PhaseConfig.PHASE_3_1 and self.total_games >= PhaseConfig.PHASE_3_1_END:
            self._save_checkpoint("p3.1_final")
            self._setup_phase(PhaseConfig.PHASE_3_2)
        elif self.phase == PhaseConfig.PHASE_3_2 and self.total_games >= PhaseConfig.PHASE_3_2_END:
            self._save_checkpoint("p3.2_final")
            self._setup_phase(PhaseConfig.PHASE_3_3)

    def _save_checkpoint(self, tag):
        path = self.checkpoint_dir / f"tessera_v5_{tag}_game{self.total_games}.pth"
        torch.save({'model': self.model.state_dict(), 'opt': self.opt.state_dict(),
                    'phase': self.phase, 'games': self.total_games}, path)
        self._log(f"Saved: {path.name}")

    def train_step(self, moves, board, policy):
        self.model.train()
        self.opt.zero_grad()
        pred, = self.model(moves, board, return_value=False)
        loss = nn.functional.cross_entropy(pred, policy.argmax(dim=1))
        loss.backward()
        if self.phase == PhaseConfig.PHASE_3_1 and not self._verify_mamba_frozen():
            raise RuntimeError("Mamba gradient leak!")
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return loss.item()

    def train_game(self, data):
        loss = self.train_step(data['moves'].to(self.device), data['board'].to(self.device), data['policy'].to(self.device))
        self.total_games += 1
        self._update_phase()
        return {'loss': loss, 'phase': self.phase, 'games': self.total_games}

    def status(self):
        return {'phase': self.phase, 'games': self.total_games, 'mamba_frozen': not any(p.requires_grad for p in self.model.mamba.parameters())}

def test_trainer():
    print("=" * 50)
    print("TesseraTrainerPhase3 Test")
    print("=" * 50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TesseraModel(vocab_size=364, d_model=256, n_layers=4)
    model.load_phase2_weights('/app/checkpoints/tessera_v4_game802591_elo1490.pth')
    trainer = TesseraTrainerPhase3(model, device=device)
    print(f"Status: {trainer.status()}")
    B, T = 4, 50
    data = {
        'moves': torch.randint(0, 363, (B, T)),
        'board': torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE)),
        'policy': torch.softmax(torch.randn(B, 364), dim=1)
    }
    result = trainer.train_game(data)
    print(f"Result: {result}")
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.mamba.parameters())
    print(f"Mamba grad check: {'FAIL' if has_grad else 'PASS'}")
    print("=" * 50)

if __name__ == "__main__":
    test_trainer()
