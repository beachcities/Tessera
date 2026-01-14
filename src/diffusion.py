"""
Tessera Phase III POC-2: Diffusion Engine
==========================================
2場（Φ_B, Φ_W）+ コウ残滓場（K）の拡散エンジン

物理場モデル:
- Φ_B: 黒石の影響力ポテンシャル
- Φ_W: 白石の影響力ポテンシャル
- K: コウ残滓場（時間的歪み）

更新則:
  Φ^{t+1}(x) = (1-α)Φ^{t}(x) + α・mean_{y∈N(x)}Φ^{t}(y) + β・S(x)
  K_{t+1}(x) = ρ・K_t(x) + γ・𝟙_{ko-at(x,t)}

Usage:
    python src/diffusion.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import time
import gc

BOARD_SIZE = 19


class DiffusionEngine(nn.Module):
    """
    2場拡散エンジン
    
    Args:
        steps: 拡散ステップ数
        alpha: 拡散率（0.1〜0.3）
        beta: ソース注入強度
        rho: コウ残滓減衰率（0.5〜0.8）
        use_diagonal: 対角近傍も使うか
        snapshot_steps: スナップショットを取るステップ（例: [2, 5, 10]）
    """
    
    def __init__(
        self,
        steps: int = 10,
        alpha: float = 0.2,
        beta: float = 1.0,
        rho: float = 0.7,
        use_diagonal: bool = True,
        snapshot_steps: Optional[List[int]] = None,
    ):
        super().__init__()
        self.steps = steps
        self.rho = rho
        self.use_diagonal = use_diagonal
        self.snapshot_steps = snapshot_steps or [2, 5, 10]
        
        # 学習可能パラメータ
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(1.0))  # 相互干渉強度
        
        # 拡散カーネル（4近傍 or 8近傍）
        self._build_kernel(use_diagonal)
    
    def _build_kernel(self, use_diagonal: bool):
        """拡散カーネルを構築"""
        if use_diagonal:
            # 8近傍（対角含む）
            kernel = torch.tensor([
                [1/8, 1/8, 1/8],
                [1/8,   0, 1/8],
                [1/8, 1/8, 1/8],
            ], dtype=torch.float32)
        else:
            # 4近傍
            kernel = torch.tensor([
                [  0, 1/4,   0],
                [1/4,   0, 1/4],
                [  0, 1/4,   0],
            ], dtype=torch.float32)
        
        # [1, 1, 3, 3] に reshape してバッファ登録
        kernel = kernel.view(1, 1, 3, 3)
        self.register_buffer('kernel', kernel)
    
    def _diffuse_step(
        self,
        field: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        """
        1ステップの拡散
        
        Args:
            field: [B, 1, 19, 19] 現在のポテンシャル場
            source: [B, 1, 19, 19] ソース項（石の配置）
        
        Returns:
            new_field: [B, 1, 19, 19] 更新後のポテンシャル場
        """
        # 近傍平均
        neighbor_mean = F.conv2d(field, self.kernel, padding=1)
        
        # 拡散更新
        alpha = torch.sigmoid(self.alpha)  # 0-1 に制限
        beta = F.softplus(self.beta)  # 正値に制限
        
        new_field = (1 - alpha) * field + alpha * neighbor_mean + beta * source
        
        return new_field
    
    def forward(
        self,
        board: torch.Tensor,
        k_field: Optional[torch.Tensor] = None,
        ko_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            board: [B, 19, 19] 盤面（1:黒, -1:白, 0:空）
            k_field: [B, 1, 19, 19] 前回のコウ残滓場（None なら初期化）
            ko_positions: [B, 19, 19] コウ発生位置（bool）
        
        Returns:
            phi_b: [B, C, 19, 19] 黒ポテンシャル場（Cはスナップショット数）
            phi_w: [B, C, 19, 19] 白ポテンシャル場
            k_field_new: [B, 1, 19, 19] 更新後のコウ残滓場
        """
        B = board.shape[0]
        device = board.device
        
        # ソース項の分離
        b_source = torch.clamp(board, min=0).unsqueeze(1)   # [B, 1, 19, 19]
        w_source = torch.clamp(-board, min=0).unsqueeze(1)  # [B, 1, 19, 19]
        
        # 初期化
        b_field = b_source.clone()
        w_field = w_source.clone()
        
        if k_field is None:
            k_field = torch.zeros(B, 1, BOARD_SIZE, BOARD_SIZE, device=device)
        
        # スナップショット収集
        b_snapshots = []
        w_snapshots = []
        
        # 拡散イテレーション
        for t in range(1, self.steps + 1):
            # 拡散ステップ
            b_field = self._diffuse_step(b_field, b_source)
            w_field = self._diffuse_step(w_field, w_source)
            
            # 相互干渉（オプション）
            gamma = torch.sigmoid(self.gamma)
            b_field = b_field - gamma * w_field * b_source.bool().float()
            w_field = w_field - gamma * b_field * w_source.bool().float()
            
            # 正値制限
            b_field = F.relu(b_field)
            w_field = F.relu(w_field)
            
            # スナップショット
            if t in self.snapshot_steps:
                b_snapshots.append(b_field)
                w_snapshots.append(w_field)
        
        # スナップショットを結合 [B, C, 19, 19]
        phi_b = torch.cat(b_snapshots, dim=1)
        phi_w = torch.cat(w_snapshots, dim=1)
        
        # コウ残滓場の更新
        k_field_new = self.rho * k_field
        if ko_positions is not None:
            k_field_new = k_field_new + ko_positions.unsqueeze(1).float()
        
        return phi_b, phi_w, k_field_new
    
    def extra_repr(self) -> str:
        return (
            f"steps={self.steps}, "
            f"snapshot_steps={self.snapshot_steps}, "
            f"use_diagonal={self.use_diagonal}"
        )


class TesseractField(nn.Module):
    """
    Tesseract Field Module: 拡散 + ray-cast の統合
    
    拡散エンジンで物理場を生成し、ray-cast で長距離認識を行う。
    """
    
    def __init__(
        self,
        diffusion_steps: int = 10,
        snapshot_steps: List[int] = [2, 5, 10],
        ray_c_out: int = 16,
    ):
        super().__init__()
        
        self.diffusion = DiffusionEngine(
            steps=diffusion_steps,
            snapshot_steps=snapshot_steps,
        )
        
        # ray-cast は後で統合（POC-2 では拡散のみテスト）
        self.c_in = len(snapshot_steps) * 2  # phi_b + phi_w
        self.ray_c_out = ray_c_out
    
    def forward(
        self,
        board: torch.Tensor,
        k_field: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        順伝播
        
        Args:
            board: [B, 19, 19] 盤面
            k_field: [B, 1, 19, 19] コウ残滓場
        
        Returns:
            field: [B, C, 19, 19] 統合ポテンシャル場
            k_field_new: [B, 1, 19, 19] 更新後コウ残滓場
        """
        phi_b, phi_w, k_field_new = self.diffusion(board, k_field)
        
        # 黒と白を結合
        field = torch.cat([phi_b, phi_w], dim=1)
        
        return field, k_field_new


def test_diffusion_basic():
    """基本動作テスト"""
    print("\n" + "=" * 60)
    print("DiffusionEngine 基本テスト")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    engine = DiffusionEngine(steps=10, snapshot_steps=[2, 5, 10]).to(device)
    print(engine)
    
    # テスト入力（ランダム盤面）
    board = torch.zeros(4, BOARD_SIZE, BOARD_SIZE, device=device)
    board[:, 9, 9] = 1    # 中央に黒石
    board[:, 9, 10] = -1  # 右に白石
    board[:, 3, 3] = 1    # 隅に黒石
    board[:, 15, 15] = -1 # 反対隅に白石
    
    phi_b, phi_w, k_field = engine(board)
    
    print(f"\n  入力 board: {board.shape}")
    print(f"  出力 phi_b: {phi_b.shape}")
    print(f"  出力 phi_w: {phi_w.shape}")
    print(f"  出力 k_field: {k_field.shape}")
    print(f"  phi_b 範囲: [{phi_b.min():.4f}, {phi_b.max():.4f}]")
    print(f"  phi_w 範囲: [{phi_w.min():.4f}, {phi_w.max():.4f}]")
    
    return True


def test_gradient_flow():
    """勾配フローテスト"""
    print("\n" + "=" * 60)
    print("勾配フローテスト")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    engine = DiffusionEngine(steps=10).to(device)
    board = torch.randn(4, BOARD_SIZE, BOARD_SIZE, device=device, requires_grad=True)
    board = torch.tanh(board)  # -1 to 1
    
    phi_b, phi_w, k_field = engine(board)
    loss = phi_b.sum() + phi_w.sum()
    loss.backward()
    
    grad_ok = board.grad is not None and not torch.isnan(board.grad).any()
    alpha_grad_ok = engine.alpha.grad is not None
    
    print(f"  入力勾配: {'✅' if grad_ok else '❌'}")
    print(f"  alpha 勾配: {'✅' if alpha_grad_ok else '❌'}")
    
    return grad_ok and alpha_grad_ok


def benchmark_diffusion(
    batch_sizes: List[int] = [16, 32, 64, 128, 256],
    steps_list: List[int] = [5, 10, 20],
    device: str = 'cuda',
):
    """ベンチマーク"""
    print("\n" + "=" * 60)
    print("DiffusionEngine ベンチマーク")
    print("=" * 60)
    
    for steps in steps_list:
        print(f"\n  Steps = {steps}:")
        engine = DiffusionEngine(steps=steps, snapshot_steps=[steps//2, steps]).to(device)
        
        for B in batch_sizes:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                board = torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device)
                board = torch.sign(board) * (torch.abs(board) > 0.5).float()
                
                # ウォームアップ
                for _ in range(3):
                    _ = engine(board)
                torch.cuda.synchronize()
                
                # 計測
                start = time.perf_counter()
                for _ in range(20):
                    phi_b, phi_w, k_field = engine(board)
                torch.cuda.synchronize()
                
                elapsed = (time.perf_counter() - start) / 20 * 1000
                peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
                
                print(f"    Batch={B:4d}: {elapsed:6.2f} ms | VRAM={peak_vram:.2f} GB ✅")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"    Batch={B:4d}: ❌ OOM")
                    torch.cuda.empty_cache()
                else:
                    raise


def test_physical_behavior():
    """物理的挙動テスト（拡散の可視化）"""
    print("\n" + "=" * 60)
    print("物理的挙動テスト")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    engine = DiffusionEngine(steps=10, snapshot_steps=[1, 5, 10]).to(device)
    
    # 単一黒石
    board = torch.zeros(1, BOARD_SIZE, BOARD_SIZE, device=device)
    board[0, 9, 9] = 1  # 中央に黒石
    
    phi_b, phi_w, _ = engine(board)
    
    print(f"\n  中央黒石からの拡散:")
    print(f"  スナップショット数: {phi_b.shape[1]}")
    
    # 各スナップショットでの中央からの減衰を確認
    for i, t in enumerate([1, 5, 10]):
        center_val = phi_b[0, i, 9, 9].item()
        neighbor_val = phi_b[0, i, 9, 10].item()
        far_val = phi_b[0, i, 9, 14].item()
        
        print(f"  t={t:2d}: 中央={center_val:.3f}, 隣接={neighbor_val:.3f}, 遠方(+5)={far_val:.3f}")
    
    # 減衰が適切か（中央 > 隣接 > 遠方）
    final = phi_b[0, -1]
    decay_ok = final[9, 9] > final[9, 10] > final[9, 14]
    print(f"\n  距離減衰: {'✅ 正常' if decay_ok else '⚠️ 要確認'}")
    
    return decay_ok


if __name__ == "__main__":
    print("=" * 60)
    print("Tessera Phase III POC-2: Diffusion Engine")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # 基本テスト
    test_diffusion_basic()
    
    # 勾配テスト
    grad_ok = test_gradient_flow()
    
    # 物理的挙動テスト
    physics_ok = test_physical_behavior()
    
    # ベンチマーク
    if device == 'cuda':
        benchmark_diffusion()
    
    # 結果サマリー
    print("\n" + "=" * 60)
    print("POC-2 基本テスト結果")
    print("=" * 60)
    print(f"  基本動作: ✅")
    print(f"  勾配フロー: {'✅' if grad_ok else '❌'}")
    print(f"  物理的挙動: {'✅' if physics_ok else '⚠️'}")
    print("=" * 60)
