"""
Tessera POC-1b v2: 極限境界テスト
=================================
本当の OOM を見つける
"""

import torch
import torch.nn as nn
import gc
import sys

sys.path.insert(0, '/app/src')
from ray_cast_v2 import RayCastLayerV2
from ray_index import BOARD_SIZE


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_extreme():
    print("=" * 70)
    print("極限境界テスト: 真の OOM を探す")
    print("=" * 70)
    
    device = 'cuda'
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU VRAM: {gpu_mem:.1f} GB")
    
    # ========================================
    # テスト 1: Batch Size 極限
    # ========================================
    print("\n[1] Batch Size 極限 (c_in=3, c_out=8)")
    
    for B in [8192, 16384, 32768, 65536]:
        clear_gpu()
        try:
            layer = RayCastLayerV2(c_in=3, c_out=8).to(device)
            x = torch.randn(B, 3, BOARD_SIZE, BOARD_SIZE, device=device)
            out = layer(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Batch={B:6d}: VRAM={peak:.2f} GB ✅")
            
            del layer, x, out, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch={B:6d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 2: 複数レイヤー同時（実際の使用に近い）
    # ========================================
    print("\n[2] 複数レイヤー同時保持 (Batch=128)")
    
    for num_layers in [1, 2, 4, 8, 16, 32]:
        clear_gpu()
        try:
            layers = nn.ModuleList([
                RayCastLayerV2(c_in=3, c_out=16) for _ in range(num_layers)
            ]).to(device)
            
            x = torch.randn(128, 3, BOARD_SIZE, BOARD_SIZE, device=device)
            
            # 連続適用
            out = x
            for layer in layers:
                out = layer(out[:, :3, :, :])  # c_in=3 に合わせる
            
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            params = sum(p.numel() for p in layers.parameters())
            print(f"  Layers={num_layers:2d}: params={params:,} | VRAM={peak:.2f} GB ✅")
            
            del layers, x, out, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Layers={num_layers:2d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 3: Mamba モデルとの同時ロード
    # ========================================
    print("\n[3] 既存 Mamba モデル + RayCast 同時ロード (Batch=128)")
    
    try:
        # Mamba モデルをロード
        from model import MambaModel
        mamba = MambaModel().to(device)
        mamba_params = sum(p.numel() for p in mamba.parameters())
        
        clear_gpu()
        torch.cuda.reset_peak_memory_stats()
        
        # 両方同時に
        mamba = MambaModel().to(device)
        raycast = RayCastLayerV2(c_in=3, c_out=16).to(device)
        
        x_mamba = torch.randint(0, 362, (128, 50), device=device)
        x_ray = torch.randn(128, 3, BOARD_SIZE, BOARD_SIZE, device=device)
        
        # Mamba forward
        out_mamba = mamba(x_mamba)
        
        # RayCast forward
        out_ray = raycast(x_ray)
        
        # 合計 loss
        loss = out_mamba.sum() + out_ray.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Mamba ({mamba_params:,} params) + RayCast: VRAM={peak:.2f} GB ✅")
        
        del mamba, raycast, x_mamba, x_ray, out_mamba, out_ray, loss
        
    except Exception as e:
        print(f"  Mamba + RayCast: {e}")
    
    # ========================================
    # テスト 4: 入力サイズを大きく（c_in 極限）
    # ========================================
    print("\n[4] 入力チャンネル極限 (Batch=128)")
    
    for c_in in [32, 64, 128, 256, 512]:
        clear_gpu()
        try:
            layer = RayCastLayerV2(c_in=c_in, c_out=32).to(device)
            x = torch.randn(128, c_in, BOARD_SIZE, BOARD_SIZE, device=device)
            out = layer(x)
            loss = out.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  c_in={c_in:3d}: VRAM={peak:.2f} GB ✅")
            
            del layer, x, out, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  c_in={c_in:3d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    print("\n" + "=" * 70)
    print("極限境界テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_extreme()
