"""
Tessera Phase III: Tessera Model
===================================
Mamba + Diffusion + RayCast + Fusion の統合モデル

アーキテクチャ:
1. 盤面 → Diffusion → 物理場スナップショット
2. 物理場 → RayCast → 長距離ポテンシャル
3. 手順 → Mamba → 時系列特徴量
4. Mamba出力 + RayCast出力 → Fusion → Policy/Value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any

# 既存コンポーネント
from model import MambaModel
from diffusion import DiffusionEngine, BOARD_SIZE
from ray_cast_v2 import RayCastLayerV2
from fusion_layer import TesseraFusion


class TesseraModel(nn.Module):
    """
    Phase III 統合モデル
    
    Phase II の MambaModel を継承しつつ、
    Tessera Field（Diffusion + RayCast）を統合
    """
    
    def __init__(
        self,
        # Mamba 設定（Phase II 互換）
        vocab_size: int = 364,
        d_model: int = 256,
        n_layers: int = 4,
        # Diffusion 設定
        diffusion_steps: int = 10,
        snapshot_steps: list = [2, 5, 10],
        # RayCast 設定
        ray_c_out: int = 16,
        ray_init_scale: float = 0.1,
        # 距離減衰（実験条件）
        distance_decay: float = 1.0,  # 1.0 = 1/d, 2.0 = 1/d²
    ):
        super().__init__()
        
        self.d_model = d_model
        self.distance_decay = distance_decay
        
        # === Phase II コンポーネント ===
        self.mamba = MambaModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
        )
        
        # === Phase III コンポーネント ===
        # 拡散エンジン
        self.diffusion = DiffusionEngine(
            steps=diffusion_steps,
            snapshot_steps=snapshot_steps,
        )
        
        # RayCast（スナップショット数 × 2 = c_in）
        c_in = len(snapshot_steps) * 2  # phi_b + phi_w
        self.raycast = RayCastLayerV2(
            c_in=c_in,
            c_out=ray_c_out,
            init_decay=distance_decay,
        )
        
        # Fusion レイヤー
        self.fusion = TesseraFusion(
            mamba_ch=d_model,
            ray_ch=ray_c_out,
            ray_init_scale=ray_init_scale,
        )
        
        # Policy Head（Fusion 後）
        self.policy_head = nn.Linear(d_model, vocab_size)
        
        # Value Head（オプション）
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )
    
    def forward(
        self,
        move_sequence: torch.Tensor,
        board: Optional[torch.Tensor] = None,
        return_value: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        順伝播
        
        Args:
            move_sequence: [B, T] 手順シーケンス（0-361: 座標, 362: パス）
            board: [B, 19, 19] 現在の盤面（None なら move_sequence から復元）
            return_value: Value も返すか
        
        Returns:
            policy: [B, vocab_size] 着手確率
            value: [B, 1] 局面評価（return_value=True の場合）
        """
        B = move_sequence.shape[0]
        device = move_sequence.device
        
        # === Mamba: 時系列処理 ===
        # MambaModel の内部で embedding + Mamba layers
        mamba_logits = self.mamba(move_sequence)  # [B, T, vocab_size]
        
        # 最後の位置の特徴量を取得
        # MambaModel の内部構造にアクセスする必要があるため、
        # ここでは簡略化して logits から逆算
        # TODO: MambaModel を改修して中間特徴量を返すようにする
        
        # 暫定: Mamba の出力を特徴量として使用
        mamba_features = mamba_logits[:, -1, :self.d_model]  # [B, d_model]
        mamba_features = mamba_features.view(B, self.d_model, 1, 1)
        mamba_features = mamba_features.expand(-1, -1, BOARD_SIZE, BOARD_SIZE)  # [B, d_model, 19, 19]
        
        # === Tessera Field: 空間物理処理 ===
        if board is not None:
            # 拡散
            phi_b, phi_w, _ = self.diffusion(board)  # [B, 3, 19, 19] each
            field = torch.cat([phi_b, phi_w], dim=1)  # [B, 6, 19, 19]
            
            # RayCast
            ray_features = self.raycast(field)  # [B, ray_c_out, 19, 19]
        else:
            # board がない場合はゼロ（Phase II 互換モード）
            ray_features = torch.zeros(
                B, self.fusion.ray_ch, BOARD_SIZE, BOARD_SIZE,
                device=device
            )
        
        # === Fusion: 統合 ===
        fused = self.fusion(mamba_features, ray_features)  # [B, d_model, 19, 19]
        
        # Global Average Pooling
        fused_pooled = fused.mean(dim=(2, 3))  # [B, d_model]
        
        # === Policy Head ===
        policy = self.policy_head(fused_pooled)  # [B, vocab_size]
        
        if return_value:
            value = self.value_head(fused_pooled)  # [B, 1]
            return policy, value
        
        return policy,
    
    def load_phase2_weights(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Phase II のチェックポイントを読み込む
        
        Args:
            checkpoint_path: Phase II のチェックポイントパス
        
        Returns:
            info: 読み込み情報
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Phase II の state_dict を取得
        if 'model_state_dict' in checkpoint:
            phase2_state = checkpoint['model_state_dict']
        else:
            phase2_state = checkpoint
        
        # Mamba 部分のみ読み込み（プレフィックス調整）
        mamba_state = {}
        for k, v in phase2_state.items():
            # Phase II のキーに mamba. プレフィックスを追加
            if not k.startswith('mamba.'):
                new_key = 'mamba.' + k
            else:
                new_key = k
            mamba_state[new_key] = v
        
        # strict=False で部分的に読み込み
        missing, unexpected = self.load_state_dict(mamba_state, strict=False)
        
        return {
            'loaded_keys': len(mamba_state),
            'missing_keys': len(missing),
            'unexpected_keys': len(unexpected),
            'phase2_elo': checkpoint.get('elo', 'unknown'),
            'phase2_games': checkpoint.get('total_games', 'unknown'),
        }
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, distance_decay={self.distance_decay}"


def test_tesseract_model():
    """統合モデルの動作確認"""
    print("=" * 60)
    print("TesseraModel 動作確認")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # モデル作成
    model = TesseraModel(
        d_model=256,
        n_layers=4,
        diffusion_steps=10,
        snapshot_steps=[2, 5, 10],
        ray_c_out=16,
        distance_decay=1.0,  # 1/d
    ).to(device)
    
    print(f"\nモデル構造:")
    print(model)
    
    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    mamba_params = sum(p.numel() for p in model.mamba.parameters())
    diffusion_params = sum(p.numel() for p in model.diffusion.parameters())
    raycast_params = sum(p.numel() for p in model.raycast.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    
    print(f"\nパラメータ数:")
    print(f"  Mamba:     {mamba_params:,}")
    print(f"  Diffusion: {diffusion_params:,}")
    print(f"  RayCast:   {raycast_params:,}")
    print(f"  Fusion:    {fusion_params:,}")
    print(f"  合計:      {total_params:,}")
    
    # テスト入力
    B = 4
    T = 50
    move_seq = torch.randint(0, 362, (B, T), device=device)
    board = torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device))
    
    print(f"\n入力:")
    print(f"  move_sequence: {move_seq.shape}")
    print(f"  board: {board.shape}")
    
    # Forward
    policy, value = model(move_seq, board, return_value=True)
    
    print(f"\n出力:")
    print(f"  policy: {policy.shape}")
    print(f"  value: {value.shape}")
    
    # Backward
    loss = policy.sum() + value.sum()
    loss.backward()
    
    print(f"\n勾配: ✅")
    
    # VRAM
    if device == 'cuda':
        vram = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"VRAM: {vram:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✅ TesseraModel 動作確認完了")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    test_tesseract_model()


def test_load_phase2():
    """Phase II チェックポイント読み込みテスト"""
    print("\n" + "=" * 60)
    print("Phase II チェックポイント読み込みテスト")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # モデル作成
    model = TesseraModel().to(device)
    
    # Phase II チェックポイント読み込み
    checkpoint_path = "/app/checkpoints/tessera_v4_game802591_elo1490.pth"
    
    try:
        info = model.load_phase2_weights(checkpoint_path)
        print(f"\n読み込み結果:")
        print(f"  読み込んだキー数: {info['loaded_keys']}")
        print(f"  欠落キー数: {info['missing_keys']}")
        print(f"  予期しないキー数: {info['unexpected_keys']}")
        print(f"  Phase II ELO: {info['phase2_elo']}")
        print(f"  Phase II ゲーム数: {info['phase2_games']}")
        print("\n✅ Phase II 重み読み込み成功")
        return True
    except Exception as e:
        print(f"\n❌ エラー: {e}")
        return False


if __name__ == "__main__":
    test_tesseract_model()
    test_load_phase2()
