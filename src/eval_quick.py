import sys
sys.path.insert(0, '/app/src')
import torch
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, PAD_TOKEN
from tessera_model import TesseraModel

def quick_eval(num_games=64, checkpoint_path=None):
    device = 'cuda'
    if checkpoint_path is None:
        checkpoint_path = '/app/checkpoints/tessera_phase3.2_fixed_final_loss5.91.pth'
    
    model = TesseraModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    engine = GPUGoEngine(batch_size=num_games, device=device)
    history = [[] for _ in range(num_games)]
    
    print(f"Evaluating {num_games} games (Model=Black vs Random)...")
    
    with torch.no_grad():
        for move_num in range(120):
            is_model_turn = (move_num % 2 == 0)
            
            if is_model_turn:
                # 黒番固定：黒(0) - 白(1)
                boards = engine.boards[:, 0] - engine.boards[:, 1]
                
                seq_tensors = []
                for h in history:
                    s = torch.tensor(h[-256:], device=device, dtype=torch.long)
                    pad = torch.full((256 - len(s),), PAD_TOKEN, device=device, dtype=torch.long)
                    seq_tensors.append(torch.cat([pad, s]))
                
                logits, = model(torch.stack(seq_tensors), boards, return_value=False)
                logits_legal = logits[:, :362]  # PASS(361)を含める
                
                legal = engine.get_legal_mask()
                logits_legal[~legal] = float('-inf')
                moves = logits_legal.argmax(dim=-1)
            else:
                legal = engine.get_legal_mask()
                moves = torch.multinomial(legal.float(), 1).squeeze(1)
            
            engine.play_batch(moves)
            for i, m in enumerate(moves.tolist()):
                history[i].append(m)
            
            if engine.is_game_over().all():
                break

    b_score, w_score, winners = engine.compute_score()
    wins = (winners > 0).sum().item()
    win_rate = wins / num_games * 100
    
    print(f"\n{'='*50}")
    print(f"Model(Black) Win Rate: {win_rate:.1f}% ({wins}/{num_games})")
    print(f"Avg Score: B:{b_score.mean():.1f} W:{w_score.mean():.1f}")
    print(f"{'='*50}\n")
    
    return wins / num_games

if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quick_eval(num_games=num_games)
