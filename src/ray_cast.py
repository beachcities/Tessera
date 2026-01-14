"""
Tessera Phase III POC-1b: Ray Cast Module
==========================================
8方向 ray-cast による長距離ポテンシャル計算。

インデックステーブルを使って gather し、
方向・距離ごとの重みで集約する。

Usage:
    python src/ray_cast.py  # テスト実行
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import time

from ray_index import RayIndexBuffer, BOARD_SIZE, MAX_DISTANCE, LINE_GROUP, DIAG_GROUP


class RayCastLayer(nn.Module):
    """
    8方向 ray-cast による長距離ポテンシャル計算レイヤー。
    
    方式A: index table + gather
    
    Args:
        max_dist: 最大距離（デフォルト: 18）
        use_d8_symmetry: D8対称性を使うか（直線/斜めで重み共有）
        init_decay: 距離減衰の初期値（1/d^init_decay）
    """
    
    def __init__(
        self,
        max_dist: int = MAX_DISTANCE,
        use_d8_symmetry: bool = True,
        init_decay: float = 1.0,
    ):
        super().__init__()
        self.max_dist = max_dist
        self.use_d8_symmetry = use_d8_symmetry
        
        # インデックスバッファ
        self.index_buffer = RayIndexBuffer(max_dist)
        
        # 重みパラメータ
        if use_d8_symmetry:
            # 直線と斜めで別々の重み [2, max_dist]
            self.weight = nn.Parameter(torch.zeros(2, max_dist))
        else:
            # 8方向それぞれ独立 [8, max_dist]
            self.weight = nn.Parameter(torch.zeros(8, max_dist))
        
        # 初期化: 距離減衰バイアス
        self._init_weights(init_decay)
    
    def _init_weights(self, decay: float):
        """距離減衰で初期化"""
        with torch.no_grad():
            distances = torch.arange(1, self.max_dist + 1, dtype=torch.float32)
            decay_values = 1.0 / (distances ** decay)
            
            if self.use_d8_symmetry:
                self.weight[0] = decay_values  # 直線
                self.weight[1] = decay_values * 0.7  # 斜め（少し弱め）
            else:
                for i in range(8):
                    if i in LINE_GROUP:
                        self.weight[i] = decay_values
                    else:
                        self.weight[i] = decay_values * 0.7
    
    def _expand_weights(self) -> torch.Tensor:
        """D8対称性を展開して [8, max_dist] の重みを返す"""
        if self.use_d8_symmetry:
            expanded = torch.zeros(8, self.max_dist, device=self.weight.device)
            expanded[LINE_GROUP] = self.weight[0]
            expanded[DIAG_GROUP] = self.weight[1]
            return expanded
        else:
            return self.weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播
        
        Args:
            x: [B, C, 19, 19] - 入力フィールド
        
        Returns:
            out: [B, C, 19, 19] - 長距離ポテンシャル
        """
        B, C, H, W = x.shape
        assert H == BOARD_SIZE and W == BOARD_SIZE, f"Expected 19x19, got {H}x{W}"
        
        # インデックスとマスクを取得
        flat_index, valid_mask = self.index_buffer.get_indices()
        # flat_index: [8, max_dist, 19, 19]
        # valid_mask: [8, max_dist, 19, 19]
        
        # 入力をフラット化 [B, C, 361]
        x_flat = x.reshape(B, C, -1)
        
        # gather のためにインデックスを拡張
        # [8, max_dist, 19, 19] -> [B, C, 8, max_dist, 361]
        idx_expanded = flat_index.view(1, 1, 8, self.max_dist, H * W)
        idx_expanded = idx_expanded.expand(B, C, -1, -1, -1)
        
        # x_flat を gather 用に拡張
        # [B, C, 361] -> [B, C, 1, 1, 361]
        x_expanded = x_flat.unsqueeze(2).unsqueeze(3).expand(-1, -1, 8, self.max_dist, -1)
        
        # gather: 各方向・距離の値を取得
        # [B, C, 8, max_dist, 361]
        gathered = torch.gather(x_expanded, dim=-1, index=idx_expanded)
        
        # reshape: [B, C, 8, max_dist, 19, 19]
        gathered = gathered.view(B, C, 8, self.max_dist, H, W)
        
        # マスク適用
        mask_expanded = valid_mask.view(1, 1, 8, self.max_dist, H, W).float()
        gathered = gathered * mask_expanded
        
        # 重みを展開して適用
        weights = self._expand_weights()  # [8, max_dist]
        weights = weights.view(1, 1, 8, self.max_dist, 1, 1)
        
        weighted = gathered * weights
        
        # 方向と距離で集約 [B, C, 19, 19]
        out = weighted.sum(dim=(2, 3))
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"max_dist={self.max_dist}, "
            f"use_d8_symmetry={self.use_d8_symmetry}, "
            f"weight_shape={tuple(self.weight.shape)}"
        )


def benchmark_ray_cast(
    batch_sizes: list = [1, 4, 8, 16, 32],
    channels: int = 3,
    num_iterations: int = 100,
    device: str = 'cuda',
):
    """ベンチマーク実行"""
    print("\n" + "=" * 60)
    print("RayCastLayer ベンチマーク")
    print("=" * 60)
    
    layer = RayCastLayer(use_d8_symmetry=True).to(device)
    
    results = []
    
    for B in batch_sizes:
        x = torch.randn(B, channels, BOARD_SIZE, BOARD_SIZE, device=device)
        
        # ウォームアップ
        for _ in range(10):
            _ = layer(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 計測
        start = time.perf_counter()
        for _ in range(num_iterations):
            out = layer(x)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) / num_iterations * 1000  # ms
        
        # VRAM 計測
        if device == 'cuda':
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            _ = layer(x)
            mem_after = torch.cuda.memory_allocated()
            vram_mb = (mem_after - mem_before) / (1024 * 1024)
        else:
            vram_mb = 0
        
        results.append({
            'batch': B,
            'time_ms': elapsed,
            'vram_mb': vram_mb,
        })
        
        print(f"  Batch={B:3d}: {elapsed:.3f} ms, VRAM: {vram_mb:.2f} MB")
    
    return results


def test_gradient_flow():
    """勾配が正しく流れるかテスト"""
    print("\n" + "=" * 60)
    print("勾配フローテスト")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    layer = RayCastLayer(use_d8_symmetry=True).to(device)
    x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE, device=device, requires_grad=True)
    
    out = layer(x)
    loss = out.sum()
    loss.backward()
    
    # 勾配チェック
    grad_ok = x.grad is not None and not torch.isnan(x.grad).any()
    weight_grad_ok = layer.weight.grad is not None and not torch.isnan(layer.weight.grad).any()
    
    print(f"  入力勾配: {'✅' if grad_ok else '❌'} shape={x.grad.shape if x.grad is not None else 'None'}")
    print(f"  重み勾配: {'✅' if weight_grad_ok else '❌'} shape={layer.weight.grad.shape if layer.weight.grad is not None else 'None'}")
    
    return grad_ok and weight_grad_ok


def test_d8_symmetry():
    """D8対称性のテスト（回転で出力が対応するか）"""
    print("\n" + "=" * 60)
    print("D8 対称性テスト")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    layer = RayCastLayer(use_d8_symmetry=True).to(device)
    
    # テスト入力（非対称なパターン）
    x = torch.zeros(1, 1, BOARD_SIZE, BOARD_SIZE, device=device)
    x[0, 0, 9, 9] = 1.0  # 中央
    x[0, 0, 9, 12] = 0.5  # 中央の右
    
    out1 = layer(x)
    
    # 90度回転した入力
    x_rot = torch.rot90(x, k=1, dims=(2, 3))
    out2 = layer(x_rot)
    out2_back = torch.rot90(out2, k=-1, dims=(2, 3))
    
    # 直線方向の重みが同じなら、出力も対応するはず
    # （完全一致ではないが、対応する位置の値が近いはず）
    diff = (out1 - out2_back).abs().max().item()
    
    print(f"  回転前後の最大差分: {diff:.6f}")
    print(f"  対称性: {'✅ 概ね保存' if diff < 0.1 else '⚠️ 要確認'}")
    
    return diff < 0.1


if __name__ == "__main__":
    print("=" * 60)
    print("Tessera Phase III POC-1b: Ray Cast Module")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # レイヤー生成
    print("\n[1] RayCastLayer 生成...")
    layer = RayCastLayer(use_d8_symmetry=True).to(device)
    print(layer)
    
    # パラメータ数
    num_params = sum(p.numel() for p in layer.parameters())
    print(f"  パラメータ数: {num_params}")
    
    # 順伝播テスト
    print("\n[2] 順伝播テスト...")
    x = torch.randn(4, 3, BOARD_SIZE, BOARD_SIZE, device=device)
    out = layer(x)
    print(f"  入力: {x.shape}")
    print(f"  出力: {out.shape}")
    print(f"  出力範囲: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    # 勾配テスト
    print("\n[3] 勾配フローテスト...")
    grad_ok = test_gradient_flow()
    
    # D8対称性テスト
    print("\n[4] D8 対称性テスト...")
    sym_ok = test_d8_symmetry()
    
    # ベンチマーク
    if device == 'cuda':
        benchmark_ray_cast(batch_sizes=[1, 4, 8, 16, 32], device=device)
    
    # 総合結果
    print("\n" + "=" * 60)
    print("POC-1b 結果サマリー")
    print("=" * 60)
    print(f"  順伝播: ✅")
    print(f"  勾配フロー: {'✅' if grad_ok else '❌'}")
    print(f"  D8 対称性: {'✅' if sym_ok else '⚠️'}")
    print("=" * 60)
