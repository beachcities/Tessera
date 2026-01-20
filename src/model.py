"""
MambaModel - Tessera/MambaGo の核心

State Space Model による手順シーケンス予測

Version: 0.2.2
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


# ============================================================
# Constants
# ============================================================

VOCAB_SIZE = 364
BOARD_SIZE = 19
PASS_TOKEN = 361
PAD_TOKEN = 362
EOS_TOKEN = 363


# ============================================================
# MambaModel
# ============================================================

class MambaModel(nn.Module):
    """
    Mamba-based Go move predictor
    
    入力: 手順のトークン列 (batch, seq_len)
    出力: 次の手の確率分布 (batch, seq_len, vocab_size)
    """
    
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 d_model: int = 256,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.turn_emb = nn.Embedding(2, d_model)  # DEC-009: 0=Self, 1=Other
        
        # Mamba layers
        try:
            from mamba_ssm import Mamba
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ])
            self.use_mamba = True
        except ImportError:
            print("⚠️ mamba_ssm not found, using LSTM fallback")
            self.mamba_layers = nn.ModuleList([
                nn.LSTM(d_model, d_model, batch_first=True)
                for _ in range(n_layers)
            ])
            self.use_mamba = False
        
        # Layer norm
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.head = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, turn_seq: torch.Tensor = None, return_hidden: bool = False) -> torch.Tensor:
        h = self.embedding(x)
        if turn_seq is not None:
            h = h + self.turn_emb(turn_seq)  # DEC-009: 主観コンテキスト付与
        h = self.dropout(h)
        
        hidden_states = []
        
        for i, (mamba, norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            residual = h
            h = norm(h)
            
            if self.use_mamba:
                h = mamba(h)
            else:
                h, _ = mamba(h)
            
            h = self.dropout(h)
            h = h + residual
            
            if return_hidden:
                hidden_states.append(h)
        
        logits = self.head(h)
        
        if return_hidden:
            return logits, hidden_states[-1]
        return logits
    
    def predict_next(self, x: torch.Tensor, temperature: float = 1.0,
                     top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            last_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                top_k_values, _ = torch.topk(last_logits, top_k, dim=-1)
                threshold = top_k_values[:, -1:].expand_as(last_logits)
                last_logits = torch.where(
                    last_logits < threshold,
                    torch.full_like(last_logits, float('-inf')),
                    last_logits
                )
            
            probs = torch.softmax(last_logits, dim=-1)
            next_move = torch.multinomial(probs, num_samples=1).squeeze(-1)
            return next_move, probs
    
    def get_move_probabilities(self, x: torch.Tensor,
                               legal_mask: Optional[torch.Tensor] = None,
                               temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            last_logits = logits[:, -1, :] / temperature
            # 362トークン（盤上+パス）のみ使用
            last_logits = last_logits[:, :362]
            
            if legal_mask is not None:
                last_logits = torch.where(
                    legal_mask,
                    last_logits,
                    torch.full_like(last_logits, float('-inf'))
                )
            
            probs = torch.softmax(last_logits, dim=-1)
            return probs


class MambaStateCapture:
    """Mamba の内部状態をキャプチャ"""
    
    def __init__(self, model: MambaModel):
        self.model = model
        self.captured_states = []
        self._hooks = []
        
        for i, layer in enumerate(model.mamba_layers):
            hook = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(hook)
    
    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.captured_states.append(output[0].detach())
            else:
                self.captured_states.append(output.detach())
        return hook
    
    def get_states(self) -> list:
        return self.captured_states
    
    def get_last_state(self) -> Optional[torch.Tensor]:
        if self.captured_states:
            return self.captured_states[-1]
        return None
    
    def clear(self):
        self.captured_states = []
    
    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


if __name__ == "__main__":
    print("Testing MambaModel...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = MambaModel(vocab_size=VOCAB_SIZE, d_model=256, d_state=16, n_layers=4).to(device)
    
    print(f"\n=== Model Info ===")
    print(f"Using Mamba: {model.use_mamba}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n=== Forward Test ===")
    x = torch.randint(0, VOCAB_SIZE, (4, 32), device=device)
    print(f"Input shape: {x.shape}")
    
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    print(f"\n=== Prediction Test ===")
    next_move, probs = model.predict_next(x)
    print(f"Next move: {next_move}")
    print(f"Top 5 probs: {probs[0].topk(5)}")
    
    print(f"\n=== State Capture Test ===")
    capture = MambaStateCapture(model)
    _ = model(x)
    states = capture.get_states()
    print(f"Captured {len(states)} states")
    if states:
        print(f"Last state norm: {states[-1].norm().item():.2f}")
    capture.remove_hooks()
    
    print(f"\n=== Gradient Test ===")
    model.train()
    target = torch.randint(0, VOCAB_SIZE, (4, 32), device=device)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, VOCAB_SIZE), target.view(-1))
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    
    print("\n✅ MambaModel test passed!")
