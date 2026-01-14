"""
勾配フロー修正テスト
"""
import torch
import sys
sys.path.insert(0, '/app/src')
from diffusion import DiffusionEngine, BOARD_SIZE

def test_gradient_fixed():
    print("=" * 60)
    print("勾配フローテスト（修正版）")
    print("=" * 60)
    
    device = 'cuda'
    engine = DiffusionEngine(steps=10).to(device)
    
    # リーフテンソルとして作成
    board = torch.zeros(4, BOARD_SIZE, BOARD_SIZE, device=device, requires_grad=True)
    
    phi_b, phi_w, k_field = engine(board)
    loss = phi_b.sum() + phi_w.sum()
    loss.backward()
    
    # 勾配確認
    board_grad_ok = board.grad is not None and not torch.isnan(board.grad).any()
    alpha_grad_ok = engine.alpha.grad is not None and not torch.isnan(engine.alpha.grad).any()
    beta_grad_ok = engine.beta.grad is not None and not torch.isnan(engine.beta.grad).any()
    
    print(f"  board 勾配: {'✅' if board_grad_ok else '❌'} shape={board.grad.shape if board.grad is not None else None}")
    print(f"  alpha 勾配: {'✅' if alpha_grad_ok else '❌'} val={engine.alpha.grad.item() if engine.alpha.grad is not None else None:.6f}")
    print(f"  beta 勾配:  {'✅' if beta_grad_ok else '❌'} val={engine.beta.grad.item() if engine.beta.grad is not None else None:.6f}")
    
    return board_grad_ok and alpha_grad_ok and beta_grad_ok

if __name__ == "__main__":
    ok = test_gradient_fixed()
    print(f"\n結果: {'✅ 全て正常' if ok else '❌ 問題あり'}")
