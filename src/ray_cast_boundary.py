"""
Tessera Phase III POC-1b v2: 境界領域テスト
============================================
8GB VRAM の限界を探る

テスト項目:
1. Batch Size の限界（512, 1024, 2048, 4096）
2. チャンネル数の限界（c_out=32, 64, 128）
3. 複合条件での限界
"""

import torch
import torch.nn as nn
import time
import gc
import sys

sys.path.insert(0, '/app/src')
from ray_index import RayIndexBuffer, BOARD_SIZE, MAX_DISTANCE
from ray_cast_v2 import RayCastLayerV2


def clear_gpu():
    """GPU メモリを完全クリア"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_boundary(
    layer_fn,
    layer_name: str,
    batch_sizes: list,
    c_in: int,
    device: str = 'cuda',
):
    """境界テスト"""
    print(f"\n{'='*70}")
    print(f"境界テスト: {layer_name}")
    print(f"{'='*70}")
    
    for B in batch_sizes:
        clear_gpu()
        
        try:
            layer = layer_fn().to(device)
            x = torch.randn(B, c_in, BOARD_SIZE, BOARD_SIZE, device=device)
            
            # forward
            out = layer(x)
            torch.cuda.synchronize()
            
            # backward（勾配計算も含めた VRAM）
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"  Batch={B:5d}: Peak VRAM = {peak_vram:.2f} GB | ✅")
            
            del layer, x, out, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch={B:5d}: ❌ OOM - 限界到達")
                clear_gpu()
                return B  # 限界の Batch Size を返す
            else:
                raise e
    
    print(f"  → 限界未到達（さらに大きな値をテスト可能）")
    return None


def main():
    print("=" * 70)
    print("Tessera POC-1b v2: 境界領域テスト")
    print("=" * 70)
    
    device = 'cuda'
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\nGPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    results = {}
    
    # ========================================
    # テスト 1: Batch Size の限界（v2 標準構成）
    # ========================================
    print("\n" + "=" * 70)
    print("[1] Batch Size 限界テスト (c_in=3, c_out=8)")
    print("=" * 70)
    
    limit = test_boundary(
        layer_fn=lambda: RayCastLayerV2(c_in=3, c_out=8),
        layer_name="v2 (3→8)",
        batch_sizes=[256, 512, 1024, 2048, 4096, 8192],
        c_in=3,
    )
    results['batch_limit_3_8'] = limit
    
    # ========================================
    # テスト 2: Batch Size の限界（大規模構成）
    # ========================================
    print("\n" + "=" * 70)
    print("[2] Batch Size 限界テスト (c_in=6, c_out=16)")
    print("=" * 70)
    
    limit = test_boundary(
        layer_fn=lambda: RayCastLayerV2(c_in=6, c_out=16),
        layer_name="v2 (6→16)",
        batch_sizes=[256, 512, 1024, 2048, 4096],
        c_in=6,
    )
    results['batch_limit_6_16'] = limit
    
    # ========================================
    # テスト 3: チャンネル数の限界（Batch=128 固定）
    # ========================================
    print("\n" + "=" * 70)
    print("[3] チャンネル数限界テスト (Batch=128)")
    print("=" * 70)
    
    channel_configs = [
        (3, 32), (3, 64), (3, 128), (3, 256),
        (6, 32), (6, 64), (6, 128),
        (16, 32), (16, 64), (16, 128),
    ]
    
    for c_in, c_out in channel_configs:
        clear_gpu()
        
        try:
            layer = RayCastLayerV2(c_in=c_in, c_out=c_out).to(device)
            x = torch.randn(128, c_in, BOARD_SIZE, BOARD_SIZE, device=device)
            
            out = layer(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            params = sum(p.numel() for p in layer.parameters())
            
            print(f"  c_in={c_in:2d}, c_out={c_out:3d}: "
                  f"params={params:6,} | VRAM={peak_vram:.2f} GB | ✅")
            
            del layer, x, out, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  c_in={c_in:2d}, c_out={c_out:3d}: ❌ OOM")
                clear_gpu()
            else:
                raise e
    
    # ========================================
    # テスト 4: 複合限界テスト
    # ========================================
    print("\n" + "=" * 70)
    print("[4] 複合限界テスト (大Batch × 大チャンネル)")
    print("=" * 70)
    
    complex_configs = [
        (512, 6, 32),
        (512, 6, 64),
        (1024, 3, 32),
        (1024, 6, 32),
        (2048, 3, 16),
        (2048, 6, 16),
    ]
    
    for B, c_in, c_out in complex_configs:
        clear_gpu()
        
        try:
            layer = RayCastLayerV2(c_in=c_in, c_out=c_out).to(device)
            x = torch.randn(B, c_in, BOARD_SIZE, BOARD_SIZE, device=device)
            
            out = layer(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"  B={B:4d}, c_in={c_in:2d}, c_out={c_out:2d}: "
                  f"VRAM={peak_vram:.2f} GB | ✅")
            
            del layer, x, out, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  B={B:4d}, c_in={c_in:2d}, c_out={c_out:2d}: ❌ OOM")
                clear_gpu()
            else:
                raise e
    
    # ========================================
    # 結果サマリー
    # ========================================
    print("\n" + "=" * 70)
    print("境界領域テスト 結果サマリー")
    print("=" * 70)
    
    for key, val in results.items():
        if val:
            print(f"  {key}: OOM at Batch={val}")
        else:
            print(f"  {key}: 限界未到達")
    
    print("\n  → 8GB VRAM の実効限界を特定")
    print("=" * 70)


if __name__ == "__main__":
    main()
