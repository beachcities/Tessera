import sys
sys.path.insert(0, '/app/src')
from long_training_v5 import Config, ParallelTrainer
import time

config = Config()
config.NUM_GAMES = 50000
config.BATCH_SIZE = 32
config.LOG_INTERVAL = 500
config.ELO_EVAL_INTERVAL = 5000
config.CHECKPOINT_INTERVAL = 5000
config.MAX_MOVES_PER_GAME = 300

print('=' * 60)
print('Phase III.1 Long Training')
print('=' * 60)

trainer = ParallelTrainer(config, device='cuda')

checkpoint_path = '/app/checkpoints/tessera_v4_game802591_elo1490.pth'
try:
    info = trainer.model.load_phase2_weights(checkpoint_path)
    print(f'Phase II weights loaded: {info["loaded_keys"]} keys')
except Exception as e:
    print(f'Starting fresh: {e}')

print(f'Target: {config.NUM_GAMES} games')
print('Starting...')
start = time.time()
last_logged = 0

while trainer.stats['total_games'] < config.NUM_GAMES:
    finished, loss = trainer.step()
    games = trainer.stats['total_games']
    
    if games >= last_logged + config.LOG_INTERVAL:
        elapsed = time.time() - start
        gps = games / elapsed
        avg_loss = trainer.stats['total_loss'] / max(1, trainer.stats['loss_count'])
        eta = (config.NUM_GAMES - games) / gps / 3600
        print(f'Game {games:6d} | Loss: {avg_loss:.4f} | Speed: {gps:.2f} g/s | ETA: {eta:.1f}h')
        last_logged = games

print('Training Complete')
