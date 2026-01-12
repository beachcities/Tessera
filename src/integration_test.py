"""
Monitor + GPUGoEngine 統合テスト
"""

import torch
from monitor import TesseraMonitor, defragment_gpu_memory
from gpu_go_engine import GPUGoEngine, PASS_TOKEN

def run_integration_test():
    print("=" * 60)
    print("🧪 Integration Test: Monitor + GPUGoEngine")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 初期化
    engine = GPUGoEngine(batch_size=4, device=device)
    monitor = TesseraMonitor()
    
    # 100手のランダム対局
    NUM_MOVES = 100
    
    for move_num in range(NUM_MOVES):
        # 合法手取得（タイミング計測）
        with monitor.time('legal_mask'):
            legal = engine.get_legal_mask()
        
        # 着手選択（ダミーのforward相当）
        with monitor.time('forward'):
            legal_indices = legal[0].nonzero().squeeze(-1)
            if len(legal_indices) == 0:
                break
            rand_idx = torch.randint(0, len(legal_indices), (1,)).item()
            selected = legal_indices[rand_idx]
            moves = torch.full((4,), selected.item(), dtype=torch.long, device=device)
        
        # 着手実行
        with monitor.time('play_move'):
            engine.play_batch(moves)
        
        # 終局チェック
        if engine.is_game_over()[0]:
            print(f"\n🏁 Game ended at move {move_num + 1}")
            break
        
        # 10手ごとに記録
        if move_num % 10 == 0:
            stones = engine.count_stones()
            monitor.record(
                move_number=move_num,
                stones_on_board=int(stones[0].sum().item()),
                legal_moves_count=int(legal[0].sum().item()),
                context_length=int(engine.move_count[0].item()),
            )
        
        # 50手ごとにデフラグ
        if move_num > 0 and move_num % 50 == 0:
            defragment_gpu_memory()
            print(f"  [Move {move_num}] Defragmented GPU memory")
    
    # サマリー表示
    monitor.print_summary()
    
    # Top 5 メトリクス
    print("\n📊 Top 5 Metrics:")
    for k, v in monitor.get_top5_metrics().items():
        print(f"   {k}: {v}")
    
    # 盤面表示
    print(f"\n🎮 Final Board (batch 0):")
    print(engine.to_string(0))
    
    print("\n✅ Integration test passed!")

if __name__ == "__main__":
    run_integration_test()
