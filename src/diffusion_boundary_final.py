"""
Tessera Phase III POC-2: 全統合 Batch 限界テスト
================================================
未確認部分のみ実行
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


def test_integrated_batch_limit():
    print("=" * 70)
    print("全統合 Batch 限界テスト (Mamba + Diffusion + RayCast)")
    print("=" * 70)
    
    device = 'cuda'
    
    from model import MambaModel
    from ray_cast_v2 import RayCastLayerV2
    
    # 128 は 0.62 GB で確認済み。256 以降をテスト
    for B in [256, 512, 1024, 2048]:
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
            
            if peak > 7.0:
                print(f"  Batch={B:4d}: VRAM={peak:.2f} GB ⚠️ 7GB超過 - 実効限界")
                break
            else:
                print(f"  Batch={B:4d}: VRAM={peak:.2f} GB ✅")
            
            del mamba, diffusion, raycast
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch={B:4d}: ❌ OOM - 限界発見!")
                break
            else:
                raise
    
    print("=" * 70)


if __name__ == "__main__":
    test_integrated_batch_limit()
