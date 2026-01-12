"""
Full Integration Test: Engine + Model + Monitor
自己対局ループの完全テスト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from monitor import TesseraMonitor, defragment_gpu_memory
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, VOCAB_SIZE
from model import MambaModel, MambaStateCapture


def run_self_play_test():
    print("=" * 60)
    print("🧪 Full Integration: Self-Play with Learning")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # 初期化
    BATCH_SIZE = 4
    SEQ_LEN = 64
    
    engine = GPUGoEngine(batch_size=BATCH_SIZE, device=device)
    model = MambaModel(vocab_size=VOCAB_SIZE, d_model=256, n_layers=4).to(device)
    monitor = TesseraMonitor()
    state_capture = MambaStateCapture(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=362)
    
    print(f"\n📊 Config:")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Seq length: {SEQ_LEN}")
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    NUM_GAMES = 2
    MOVES_PER_GAME = 50
    total_loss = 0
    total_steps = 0
    
    for game in range(NUM_GAMES):
        print(f"\n🎮 Game {game + 1}/{NUM_GAMES}")
        engine.reset()
        state_capture.clear()
        
        game_moves = []
        
        for move_num in range(MOVES_PER_GAME):
            with monitor.time('legal_mask'):
                legal_mask = engine.get_legal_mask()
            
            with monitor.time('forward'):
                history = engine.get_current_sequence(max_len=SEQ_LEN)
                
                if history.shape[1] < SEQ_LEN:
                    pad = torch.full(
                        (BATCH_SIZE, SEQ_LEN - history.shape[1]),
                        362, dtype=torch.long, device=device
                    )
                    seq = torch.cat([pad, history], dim=1)
                else:
                    seq = history[:, -SEQ_LEN:]
                
                model.eval()
                with torch.no_grad():
                    probs = model.get_move_probabilities(seq, legal_mask)
                
                selected_moves = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            with monitor.time('play_move'):
                engine.play_batch(selected_moves)
            
            game_moves.append(selected_moves.clone())
            
            if engine.is_game_over().all():
                print(f"   Game ended at move {move_num + 1}")
                break
            
            if move_num % 10 == 0:
                stones = engine.count_stones()
                ssm_state = state_capture.get_last_state()
                monitor.record(
                    move_number=move_num,
                    iteration=game * MOVES_PER_GAME + move_num,
                    ssm_state=ssm_state,
                    stones_on_board=int(stones.sum().item()),
                    legal_moves_count=int(legal_mask.sum().item()),
                )
        
        if len(game_moves) > 1:
            print(f"   Learning from {len(game_moves)} moves...")
            model.train()
            
            moves_tensor = torch.stack(game_moves, dim=1)
            
            if moves_tensor.shape[1] > 1:
                input_seq = moves_tensor[:, :-1]
                target_seq = moves_tensor[:, 1:]
                
                if input_seq.shape[1] < SEQ_LEN:
                    pad_len = SEQ_LEN - input_seq.shape[1]
                    pad = torch.full((BATCH_SIZE, pad_len), 362, dtype=torch.long, device=device)
                    input_seq = torch.cat([pad, input_seq], dim=1)
                    target_pad = torch.full((BATCH_SIZE, pad_len), 362, dtype=torch.long, device=device)
                    target_seq = torch.cat([target_pad, target_seq], dim=1)
                
                optimizer.zero_grad()
                logits = model(input_seq)
                loss = criterion(logits.view(-1, VOCAB_SIZE), target_seq.view(-1))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_steps += 1
                print(f"   Loss: {loss.item():.4f}")
        
        defragment_gpu_memory()
    
    print("\n" + "=" * 60)
    monitor.print_summary()
    
    if total_steps > 0:
        print(f"\n📈 Learning Summary:")
        print(f"   Average Loss: {total_loss / total_steps:.4f}")
    
    print(f"\n🎮 Final Board (Batch 0):")
    print(engine.to_string(0))
    
    print("\n✅ Full integration test passed!")


if __name__ == "__main__":
    run_self_play_test()
