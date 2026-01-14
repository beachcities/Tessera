"""
Tessera Phase III POC-2: 拡散エンジン境界テスト
================================================
8GB VRAM での限界を探る
"""

import torch
import torch.nn as nn
import gc
import sys

sys.path.insert(0, '/app/src')
from diffusion import DiffusionEngine, TesseractField, BOARD_SIZE


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def test_boundary():
    print("=" * 70)
    print("POC-2 拡散エンジン 境界テスト")
    print("=" * 70)
    
    device = 'cuda'
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU VRAM: {gpu_mem:.1f} GB")
    
    # ========================================
    # テスト 1: Batch Size 極限
    # ========================================
    print("\n[1] Batch Size 極限 (Steps=10)")
    
    for B in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]:
        clear_gpu()
        try:
            engine = DiffusionEngine(steps=10, snapshot_steps=[2, 5, 10]).to(device)
            board = torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Batch={B:6d}: VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch={B:6d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 2: Steps 数の限界
    # ========================================
    print("\n[2] Steps 数極限 (Batch=128)")
    
    for steps in [10, 20, 50, 100, 200, 500]:
        clear_gpu()
        try:
            snapshot_steps = [steps // 4, steps // 2, steps]
            engine = DiffusionEngine(steps=steps, snapshot_steps=snapshot_steps).to(device)
            board = torch.sign(torch.randn(128, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Steps={steps:4d}: VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Steps={steps:4d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 3: スナップショット数の限界
    # ========================================
    print("\n[3] スナップショット数極限 (Batch=128, Steps=50)")
    
    for num_snapshots in [3, 6, 10, 20, 30, 50]:
        clear_gpu()
        try:
            steps = 50
            snapshot_steps = [int(steps * i / num_snapshots) for i in range(1, num_snapshots + 1)]
            engine = DiffusionEngine(steps=steps, snapshot_steps=snapshot_steps).to(device)
            board = torch.sign(torch.randn(128, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            c_out = phi_b.shape[1]
            print(f"  Snapshots={num_snapshots:2d} (c_out={c_out:2d}): VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Snapshots={num_snapshots:2d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    # ========================================
    # テスト 4: 複合条件（大Batch × 多Steps）
    # ========================================
    print("\n[4] 複合条件 (大Batch × 多Steps)")
    
    configs = [
        (512, 20),
        (1024, 20),
        (2048, 10),
        (2048, 20),
        (4096, 10),
        (4096, 20),
        (8192, 10),
    ]
    
    for B, steps in configs:
        clear_gpu()
        try:
            snapshot_steps = [steps // 2, steps]
            engine = DiffusionEngine(steps=steps, snapshot_steps=snapshot_steps).to(device)
            board = torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE, device=device))
            
            phi_b, phi_w, k_field = engine(board)
            loss = phi_b.sum() + phi_w.sum()
            loss.backward()
            torch.cuda.synchronize()
            
            peak = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  B={B:5d}, Steps={steps:2d}: VRAM={peak:.2f} GB ✅")
            
            del engine, board, phi_b, phi_w, k_field, loss
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  B={B:5d}, Steps={steps:2d}: ❌ OOM")
                clear_gpu()
            else:
                raise
    
    # ========================================
    # テスト 5: Mamba + Diffusion 同時ロード
    # ========================================
    print("\n[5] Mamba + Diffusion 同時ロード (Batch=128)")
    
    try:
        from model import MambaModel
        
        clear_gpu()
        mamba = MambaModel().to(device)
        diffusion = DiffusionEngine(steps=10, snapshot_steps=[2, 5, 10]).to(device)
        
        mamba_params = sum(p.numel() for p in mamba.parameters())
        diff_params = sum(p.numel() for p in diffusion.parameters())
        
        # Mamba 入力
        x_mamba = torch.randint(0, 362, (128, 50), device=device)
        # Diffusion 入力
        board = torch.sign(torch.randn(128, BOARD_SIZE, BOARD_SIZE, device=device))
        
        out_mamba = mamba(x_mamba)
        phi_b, phi_w, k_field = diffusion(board)
        
        loss = out_mamba.sum() + phi_b.sum() + phi_w.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Mamba ({mamba_params:,}) + Diffusion ({diff_params}): VRAM={peak:.2f} GB ✅")
        
    except Exception as e:
        print(f"  Mamba + Diffusion: {e}")
    
    # ========================================
    # テスト 6: Mamba + Diffusion + RayCast 全統合
    # ========================================
    print("\n[6] 全統合テスト: Mamba + Diffusion + RayCast (Batch=128)")
    
    try:
        from model import MambaModel
        from ray_cast_v2 import RayCastLayerV2
        
        clear_gpu()
        mamba = MambaModel().to(device)
        diffusion = DiffusionEngine(steps=10, snapshot_steps=[2, 5, 10]).to(device)
        raycast = RayCastLayerV2(c_in=6, c_out=16).to(device)  # phi_b(3) + phi_w(3) = 6
        
        total_params = (
            sum(p.numel() for p in mamba.parameters()) +
            sum(p.numel() for p in diffusion.parameters()) +
            sum(p.numel() for p in raycast.parameters())
        )
        
        # 入力
        x_mamba = torch.randint(0, 362, (128, 50), device=device)
        board = torch.sign(torch.randn(128, BOARD_SIZE, BOARD_SIZE, device=device))
        
        # Forward
        out_mamba = mamba(x_mamba)
        phi_b, phi_w, k_field = diffusion(board)
        field = torch.cat([phi_b, phi_w], dim=1)  # [B, 6, 19, 19]
        long_range = raycast(field)  # [B, 16, 19, 19]
        
        loss = out_mamba.sum() + long_range.sum()
        loss.backward()
        torch.cuda.synchronize()
        
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  全統合 ({total_params:,} params): VRAM={peak:.2f} GB ✅")
        
    except Exception as e:
        print(f"  全統合: {e}")
    
    print("\n" + "=" * 70)
    print("POC-2 境界テスト完了")
    print("=" * 70)


if __name__ == "__main__":
    test_boundary()
