"""
Tessera Phase III POC-2: 拡散エンジン極限境界テスト
====================================================
真の OOM を見つける
"""

import torch
import gc
import sys

sys.path.insert(0, '/app/src')
from diffusion import DiffusionEngine, BOARD_SIZE


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_extreme():
    print("=" * 70)
    print("POC-2 拡散エンジン 極限境界テスト")
    print("=" * 70)
    
    device = 'cuda'
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU VRAM: {gpu_mem:.1f} GB")
    
    # ========================================
    # テスト 1: Batch Size 極限（さらに大きく）
    # ========================================
    print("\n[1] Batch Size 極限 (Steps=10)")
    
    for B in [32768, 65536, 131072, 262144]:
        clear_gpu()
        try:
            engine = DiffusionEngine(steps=10, snapshot_steps=[5, 10]).to(device)
            board = torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Batch={B:7d}: VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch={B:7d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 2: Steps 極限（さらに多く）
    # ========================================
    print("\n[2] Steps 極限 (Batch=128)")
    
    for steps in [500, 1000, 2000, 5000]:
        clear_gpu()
        try:
            snapshot_steps = [steps]  # 最終のみ
            engine = DiffusionEngine(steps=steps, snapshot_steps=snapshot_steps).to(device)
            board = torch.sign(torch.randn(128, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Steps={steps:5d}: VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Steps={steps:5d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 3: 複合極限（大Batch × 多Steps × 多Snapshot）
    # ========================================
    print("\n[3] 複合極限 (大Batch × 多Steps × 多Snapshot)")
    
    configs = [
        (16384, 20, 10),
        (32768, 20, 10),
        (16384, 50, 20),
        (8192, 100, 30),
        (4096, 200, 50),
    ]
    
    for B, steps, num_snap in configs:
        clear_gpu()
        try:
            snapshot_steps = [int(steps * i / num_snap) for i in range(1, num_snap + 1)]
            engine = DiffusionEngine(steps=steps, snapshot_steps=snapshot_steps).to(device)
            board = torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  B={B:5d}, Steps={steps:3d}, Snap={num_snap:2d}: VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  B={B:5d}, Steps={steps:3d}, Snap={num_snap:2d}: ❌ OOM")
                clear_gpu()
            else:
                raise
    
    # ========================================
    # テスト 4: 全統合の Batch 限界
    # ========================================
    print("\n[4] 全統合 Batch 限界 (Mamba + Diffusion + RayCast)")
    
    try:
        from model import MambaModel
        from ray_cast_v2 import RayCastLayerV2
        
        for B in [128, 256, 512, 1024, 2048, 4096]:
            clear_gpu()
            try:
                mamba = MambaModel().to(device)
                diffusion = DiffusionEngine(steps=10, snapshot_steps=[2, 5, 10]).to(device)
                raycast = RayCastLayerV2(c_in=6, c_out=16).to(device)
                
                x_mamba = torch.randint(0, 362, (B, 50), device=device)
                board = torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device))
                
                out_mamba = mamba(x_mamba)
                phi_b, phi_w, k_field = diffusion(board)
                field = torch.cat([phi_b, phi_w], dim=1)
                long_range = raycast(field)
                
                loss = out_mamba.sum() + long_range.sum()
                loss.backward()
                torch.cuda.synchronize()
                
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                print(f"  全統合 Batch={B:4d}: VRAM={peak:.2f} GB ✅")
                
                del mamba, diffusion, raycast, x_mamba, board, out_mamba, phi_b, phi_w, k_field, field, long_range, loss
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  全統合 Batch={B:4d}: ❌ OOM - 限界発見!")
                    break
                else:
                    raise
    except ImportError as e:
        print(f"  インポートエラー: {e}")
    
    print("\n" + "=" * 70)
    print("POC-2 極限境界テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_extreme()
