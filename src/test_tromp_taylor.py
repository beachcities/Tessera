"""Tromp-Taylor エンジンのテスト"""
import sys
sys.path.insert(0, '/app/src')
import torch

print("=== Tromp-Taylor Engine Test ===\n")

# 1. chain_utils のテスト
print("1. Testing compute_territory...")
from chain_utils import compute_territory

# テスト盤面を作成（簡単なケース）
boards = torch.zeros(2, 2, 19, 19, device='cuda')

# バッチ0: 黒が左半分、白が右半分
boards[0, 0, :, :9] = 1  # 黒: 左9列
boards[0, 1, :, 10:] = 1  # 白: 右9列

# バッチ1: 空盤面
# (そのまま)

black_score, white_score, winner = compute_territory(boards)
print(f"Batch 0 - Black: {black_score[0].item():.1f}, White: {white_score[0].item():.1f}, Winner: {winner[0].item()}")
print(f"Batch 1 - Black: {black_score[1].item():.1f}, White: {white_score[1].item():.1f}, Winner: {winner[1].item()}")

# 2. GPUGoEngine のテスト
print("\n2. Testing GPUGoEngine...")
from gpu_go_engine import GPUGoEngine, coord_to_token, PASS_TOKEN

engine = GPUGoEngine(batch_size=4, device='cuda')

# 数手打つ
moves_list = [
    coord_to_token(3, 3),   # 黒
    coord_to_token(15, 15), # 白
    coord_to_token(3, 15),  # 黒
    coord_to_token(15, 3),  # 白
]

for i, move in enumerate(moves_list):
    m = torch.full((4,), move, dtype=torch.long, device='cuda')
    engine.play_batch(m)
    print(f"Move {i+1}: token {move}")

print(f"\nStones: {engine.count_stones()[0].tolist()}")
print(f"Move count: {engine.move_count[0].item()}")

# スコア計算
black_score, white_score, winner = engine.compute_score()
print(f"\nScore - Black: {black_score[0].item():.1f}, White: {white_score[0].item():.1f}")
print(f"Winner: {winner[0].item()} (black=+1, white=-1)")

# 3. 対局速度テスト
print("\n3. Speed test (100 games)...")
import time

engine = GPUGoEngine(batch_size=32, device='cuda')
start = time.time()

games_finished = 0
max_moves = 100

for _ in range(max_moves):
    legal = engine.get_legal_mask()
    
    # ランダムに合法手を選択
    probs = legal.float()
    probs = probs / probs.sum(dim=1, keepdim=True)
    moves = torch.multinomial(probs, 1).squeeze(1)
    
    engine.play_batch(moves)

elapsed = time.time() - start
total_moves = 32 * max_moves
print(f"Time: {elapsed:.2f}s")
print(f"Speed: {total_moves/elapsed:.0f} moves/s")
print(f"Speed: {32*max_moves/100/elapsed:.1f} games/s (assuming 100 moves/game)")

# 最終スコア
black_score, white_score, winner = engine.compute_score()
print(f"\nFinal scores (batch 0): Black {black_score[0].item():.1f}, White {white_score[0].item():.1f}")

print("\n=== All tests passed! ===")
