"""Phase 3.1a: 1000 games training with Mamba frozen"""
import torch
import time
from datetime import datetime
from tessera_model import TesseraModel
from trainer_phase3 import TesseraTrainerPhase3
from diffusion import BOARD_SIZE

def run_phase3_1a(num_games=1000, log_interval=100):
    print("=" * 60)
    print("Phase 3.1a: 1000 Games Training")
    print("=" * 60)
    
    device = 'cuda'
    model = TesseraModel(vocab_size=364, d_model=256, n_layers=4)
    info = model.load_phase2_weights('/app/checkpoints/tessera_v4_game802591_elo1490.pth')
    print(f"Phase II weights loaded: {info['loaded_keys']} keys")
    
    trainer = TesseraTrainerPhase3(model, device=device)
    
    losses = []
    start_time = time.time()
    
    for i in range(num_games):
        B, T = 8, 100
        data = {
            'moves': torch.randint(0, 363, (B, T)),
            'board': torch.sign(torch.randn(B, BOARD_SIZE, BOARD_SIZE)),
            'policy': torch.softmax(torch.randn(B, 364), dim=1)
        }
        result = trainer.train_game(data)
        losses.append(result['loss'])
        
        if (i + 1) % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            elapsed = time.time() - start_time
            games_per_sec = (i + 1) / elapsed
            print(f"Game {i+1}/{num_games} | Loss: {avg_loss:.4f} | {games_per_sec:.1f} g/s")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("Phase 3.1a Complete")
    print("=" * 60)
    print(f"Total games: {num_games}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss (avg last 100): {sum(losses[-100:])/100:.4f}")
    print(f"Trend: {'DOWN' if losses[-1] < losses[0] else 'UP/FLAT'}")
    
    # Save checkpoint
    trainer._save_checkpoint("p3.1a_game1000")
    
    return losses

if __name__ == "__main__":
    run_phase3_1a()
