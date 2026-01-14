"""
Tessera Phase III: Tesseract Fusion Layer
==========================================
Mamba 出力 + Tesseract Field (RayCast) の後期融合

設計思想:
- Mamba (128ch): 主 - 時系列・トポロジー記憶
- RayCast (16ch): 副 - 長距離幾何学的直感
- 学習初期は Mamba 優位、徐々に RayCast が効き始める
"""

import torch
import torch.nn as nn


class TesseraFusion(nn.Module):
    """
    Mamba 出力 + RayCast 出力の後期融合
    
    Args:
        mamba_ch: Mamba 出力チャンネル数 (default: 128)
        ray_ch: RayCast 出力チャンネル数 (default: 16)
        ray_init_scale: RayCast 側の初期スケール (default: 0.1)
    """
    
    def __init__(self, mamba_ch: int = 128, ray_ch: int = 16, ray_init_scale: float = 0.1):
        super().__init__()
        self.mamba_ch = mamba_ch
        self.ray_ch = ray_ch
        self.ray_init_scale = ray_init_scale
        
        total_ch = mamba_ch + ray_ch
        
        # GroupNorm(1, ch) = LayerNorm equivalent, no permute needed
        self.norm = nn.GroupNorm(1, total_ch)
        self.conv = nn.Conv2d(total_ch, mamba_ch, kernel_size=1)
        self.act = nn.GELU()
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初期化 + RayCast 側を小さく"""
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        
        # RayCast チャンネル（後半 16ch）の重みを小さくする
        with torch.no_grad():
            self.conv.weight[:, self.mamba_ch:, :, :] *= self.ray_init_scale
    
    def forward(self, mamba_out: torch.Tensor, ray_out: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            mamba_out: [B, 128, 19, 19] Mamba の出力
            ray_out:   [B, 16, 19, 19]  RayCast の出力
        
        Returns:
            out: [B, 128, 19, 19] 統合された特徴量
        """
        x = torch.cat([mamba_out, ray_out], dim=1)  # [B, 144, 19, 19]
        x = self.norm(x)
        x = self.act(self.conv(x))  # [B, 128, 19, 19]
        return x
    
    def extra_repr(self) -> str:
        return f"mamba_ch={self.mamba_ch}, ray_ch={self.ray_ch}, ray_init_scale={self.ray_init_scale}"


if __name__ == "__main__":
    # 動作確認
    print("=" * 60)
    print("TesseraFusion 動作確認")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fusion = TesseraFusion().to(device)
    print(fusion)
    
    # テスト入力
    mamba_out = torch.randn(4, 128, 19, 19, device=device)
    ray_out = torch.randn(4, 16, 19, 19, device=device)
    
    out = fusion(mamba_out, ray_out)
    print(f"\n入力: mamba={mamba_out.shape}, ray={ray_out.shape}")
    print(f"出力: {out.shape}")
    
    # 勾配テスト
    loss = out.sum()
    loss.backward()
    print(f"勾配: ✅")
    
    # RayCast 側の重みスケール確認
    ray_weight_norm = fusion.conv.weight[:, 128:, :, :].norm().item()
    mamba_weight_norm = fusion.conv.weight[:, :128, :, :].norm().item()
    print(f"\n重みノルム: Mamba={mamba_weight_norm:.4f}, RayCast={ray_weight_norm:.4f}")
    print(f"比率: {ray_weight_norm/mamba_weight_norm:.4f} (期待値: ~0.1)")
    
    print("\n" + "=" * 60)
    print("✅ TesseraFusion 準備完了")
    print("=" * 60)
