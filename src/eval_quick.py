import sys
sys.path.insert(0, '/app/src')
import torch
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, PAD_TOKEN
from tessera_model import TesseraModel

@torch.no_grad()
def quick_eval(model, device='cuda', num_games=64, verbose=True):
    """
    モデルを直接受け取って評価する（学習ループ組み込み用）
    
    Args:
        model: TesseraModel インスタンス（メモリ上）
        device: 'cuda' or 'cpu'
        num_games: 評価ゲーム数
        verbose: 詳細出力するか
    
    Returns:
        win_rate: 0.0 ~ 1.0
    """
    was_training = model.training
    model.eval()
    
    try:
        engine = GPUGoEngine(batch_size=num_games, device=device)
        history = [[] for _ in range(num_games)]
        
        if verbose:
            print(f"Evaluating {num_games} games (Model=Black vs Random)...")
        
        for move_num in range(120):
            is_model_turn = (move_num % 2 == 0)
            
            if is_model_turn:
                boards = engine.boards[:, 0] - engine.boards[:, 1]
                
                seq_tensors = []
                for h in history:
                    s = torch.tensor(h[-256:], device=device, dtype=torch.long)
                    pad = torch.full((256 - len(s),), PAD_TOKEN, device=device, dtype=torch.long)
                    seq_tensors.append(torch.cat([pad, s]))
                
                # DEC-009: turn_seq 生成（Model=黒番固定、偶数手=自分(0)、奇数手=相手(1)）
                turn_seq_list = []
                for h in history:
                    history_len = len(h)
                    if history_len > 0:
                        ts = torch.tensor([j % 2 for j in range(history_len)], device=device, dtype=torch.long)
                        ts = ts[-256:]
                        ts_pad = torch.zeros(256 - len(ts), device=device, dtype=torch.long)
                        turn_seq_list.append(torch.cat([ts_pad, ts]))
                    else:
                        turn_seq_list.append(torch.zeros(256, device=device, dtype=torch.long))
                turn_seq = torch.stack(turn_seq_list)
                
                logits, = model(torch.stack(seq_tensors), boards, turn_seq=turn_seq, return_value=False)
                logits_legal = logits[:, :362]
                
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
        win_rate = wins / num_games
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Model(Black) Win Rate: {win_rate*100:.1f}% ({wins}/{num_games})")
            print(f"Avg Score: B:{b_score.mean():.1f} W:{w_score.mean():.1f}")
            print(f"{'='*50}\n")
        
        return win_rate
    
    finally:
        if was_training:
            model.train()


# 後方互換性のためのラッパー（スタンドアロン実行用）
def quick_eval_from_checkpoint(num_games=64, checkpoint_path=None):
    """チェックポイントからロードして評価（スタンドアロン用）"""
    device = 'cuda'
    if checkpoint_path is None:
        checkpoint_path = '/app/checkpoints/tessera_phase3.2_fixed_final_loss5.91.pth'
    
    model = TesseraModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    
    return quick_eval(model, device, num_games, verbose=True)


if __name__ == "__main__":
    num_games = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    quick_eval_from_checkpoint(num_games=num_games)
