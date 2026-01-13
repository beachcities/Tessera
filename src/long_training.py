"""
Tessera Long Training Script
============================
放置可能な長期学習スクリプト

Features:
- 100ゲームごとにチェックポイント保存
- 10ゲームごとにログ出力
- OOM/SSM発散で自動停止
- ログファイル出力

Usage:
    python3.10 src/long_training.py

Version: 0.3.0
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import traceback
from pathlib import Path

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor import TesseraMonitor, defragment_gpu_memory
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, VOCAB_SIZE
from model import MambaModel, MambaStateCapture


# ============================================================
# Configuration
# ============================================================

class Config:
    # 学習設定
    NUM_GAMES = 20000              # 総ゲーム数
    MOVES_PER_GAME = 150         # 1ゲームあたりの最大手数
    BATCH_SIZE = 16               # バッチサイズ
    SEQ_LEN = 64                 # シーケンス長
    
    # モデル設定
    D_MODEL = 256
    N_LAYERS = 4
    LEARNING_RATE = 1e-4
    
    # ログ・保存設定
    LOG_INTERVAL = 10            # ログ出力間隔（ゲーム数）
    CHECKPOINT_INTERVAL = 100    # チェックポイント間隔（ゲーム数）
    DEFRAG_INTERVAL = 50         # デフラグ間隔（手数）
    
    # 自動停止閾値
    MAX_SSM_NORM = 200.0         # SSM State ノルム上限
    MAX_VRAM_RATIO = 0.95        # VRAM使用率上限
    
    # パス
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"


# ============================================================
# Training Functions
# ============================================================

def setup_logging(config: Config) -> str:
    """ログ設定"""
    Path(config.LOG_DIR).mkdir(exist_ok=True)
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config.LOG_DIR, f"training_{timestamp}.log")
    
    return log_file


def log(message: str, log_file: str = None, also_print: bool = True):
    """ログ出力"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    
    if also_print:
        print(line, flush=True)
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(line + "\n")


def save_checkpoint(model, optimizer, game_num, total_loss, avg_loss, filepath):
    """チェックポイント保存"""
    torch.save({
        'game': game_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'total_loss': total_loss,
        'avg_loss': avg_loss,
    }, filepath)


def check_stop_conditions(monitor: TesseraMonitor, config: Config) -> tuple[bool, str]:
    """停止条件チェック"""
    if not monitor.history:
        return False, ""
    
    latest = monitor.history[-1]
    
    # SSM State 発散
    if latest.mamba and latest.mamba.ssm_state_norm > config.MAX_SSM_NORM:
        return True, f"SSM State diverged: {latest.mamba.ssm_state_norm:.1f}"
    
    # VRAM 限界
    if latest.vram_max_mb > 0:
        vram_ratio = latest.vram_used_mb / latest.vram_max_mb
        if vram_ratio > config.MAX_VRAM_RATIO:
            return True, f"VRAM limit: {vram_ratio*100:.1f}%"
    
    return False, ""


def run_single_game(engine, model, state_capture, monitor, optimizer, criterion, 
                    config, device, game_num) -> tuple[float, int, bool]:
    """
    1ゲームを実行して学習
    
    Returns:
        loss: ゲームのロス
        moves: 手数
        early_stop: 異常停止フラグ
    """
    engine.reset()
    state_capture.clear()
    
    game_moves = []
    
    for move_num in range(config.MOVES_PER_GAME):
        # 合法手取得
        legal_mask = engine.get_legal_mask()
        
        # モデル推論
        history = engine.get_current_sequence(max_len=config.SEQ_LEN)
        
        if history.shape[1] < config.SEQ_LEN:
            pad = torch.full(
                (config.BATCH_SIZE, config.SEQ_LEN - history.shape[1]),
                362, dtype=torch.long, device=device
            )
            seq = torch.cat([pad, history], dim=1)
        else:
            seq = history[:, -config.SEQ_LEN:]
        
        model.eval()
        with torch.no_grad():
            probs = model.get_move_probabilities(seq, legal_mask)
        
        selected_moves = torch.multinomial(probs, num_samples=1).squeeze(-1)
        
        # 着手
        engine.play_batch(selected_moves)
        game_moves.append(selected_moves.clone())
        
        # 終局チェック
        if engine.is_game_over().all():
            break
        
        # 定期的にモニタリング
        if move_num % 10 == 0:
            ssm_state = state_capture.get_last_state()
            monitor.record(
                move_number=move_num,
                iteration=game_num * config.MOVES_PER_GAME + move_num,
                ssm_state=ssm_state,
                stones_on_board=int(engine.count_stones().sum().item()),
                legal_moves_count=int(legal_mask.sum().item()),
            )
            
            # 停止条件チェック
            should_stop, reason = check_stop_conditions(monitor, config)
            if should_stop:
                return 0.0, move_num, True
        
        # 定期的にデフラグ
        if move_num > 0 and move_num % config.DEFRAG_INTERVAL == 0:
            defragment_gpu_memory()
    
    # 学習
    loss_value = 0.0
    if len(game_moves) > 1:
        model.train()
        
        moves_tensor = torch.stack(game_moves, dim=1)
        
        if moves_tensor.shape[1] > 1:
            input_seq = moves_tensor[:, :-1]
            target_seq = moves_tensor[:, 1:]
            
            if input_seq.shape[1] < config.SEQ_LEN:
                pad_len = config.SEQ_LEN - input_seq.shape[1]
                pad = torch.full((config.BATCH_SIZE, pad_len), 362, dtype=torch.long, device=device)
                input_seq = torch.cat([pad, input_seq], dim=1)
                target_pad = torch.full((config.BATCH_SIZE, pad_len), 362, dtype=torch.long, device=device)
                target_seq = torch.cat([target_pad, target_seq], dim=1)
            
            optimizer.zero_grad()
            logits = model(input_seq)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            loss_value = loss.item()
    
    return loss_value, len(game_moves), False


def main():
    """メインの学習ループ"""
    config = Config()
    log_file = setup_logging(config)
    
    log("=" * 60, log_file)
    log("🚀 Tessera Long Training Started", log_file)
    log("=" * 60, log_file)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f"Device: {device}", log_file)
    
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)", log_file)
    
    # 初期化
    log("\n📦 Initializing components...", log_file)
    
    engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
    model = MambaModel(
        vocab_size=VOCAB_SIZE, 
        d_model=config.D_MODEL, 
        n_layers=config.N_LAYERS
    ).to(device)
    monitor = TesseraMonitor()
    state_capture = MambaStateCapture(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=362)
    
    num_params = sum(p.numel() for p in model.parameters())
    log(f"Model parameters: {num_params:,}", log_file)
    
    log(f"\n📊 Config:", log_file)
    log(f"   Games: {config.NUM_GAMES}", log_file)
    log(f"   Moves/game: {config.MOVES_PER_GAME}", log_file)
    log(f"   Batch size: {config.BATCH_SIZE}", log_file)
    log(f"   Seq length: {config.SEQ_LEN}", log_file)
    log(f"   Log interval: {config.LOG_INTERVAL} games", log_file)
    log(f"   Checkpoint interval: {config.CHECKPOINT_INTERVAL} games", log_file)
    
    # 学習ループ
    log("\n🎮 Starting training loop...\n", log_file)
    
    start_time = time.time()
    total_loss = 0.0
    total_steps = 0
    losses_window = []  # 直近のロスを保持
    
    try:
        for game_num in range(1, config.NUM_GAMES + 1):
            game_start = time.time()
            
            loss, moves, early_stop = run_single_game(
                engine, model, state_capture, monitor, optimizer, criterion,
                config, device, game_num
            )
            
            if early_stop:
                should_stop, reason = check_stop_conditions(monitor, config)
                log(f"\n🛑 Early stop at game {game_num}: {reason}", log_file)
                break
            
            total_loss += loss
            total_steps += 1
            losses_window.append(loss)
            if len(losses_window) > 50:
                losses_window.pop(0)
            
            game_time = time.time() - game_start
            
            # 定期ログ
            if game_num % config.LOG_INTERVAL == 0:
                avg_loss = total_loss / total_steps if total_steps > 0 else 0
                recent_loss = sum(losses_window) / len(losses_window) if losses_window else 0
                elapsed = time.time() - start_time
                games_per_hour = game_num / (elapsed / 3600) if elapsed > 0 else 0
                
                vram = monitor.snapshot_vram()
                vram_pct = vram['used_mb'] / vram['max_mb'] * 100 if vram['max_mb'] > 0 else 0
                
                log(
                    f"Game {game_num:4d}/{config.NUM_GAMES} | "
                    f"Loss: {recent_loss:.4f} (avg: {avg_loss:.4f}) | "
                    f"Moves: {moves:3d} | "
                    f"VRAM: {vram_pct:.0f}% | "
                    f"Speed: {games_per_hour:.1f} games/hr",
                    log_file
                )
            
            # チェックポイント保存
            if game_num % config.CHECKPOINT_INTERVAL == 0:
                avg_loss = total_loss / total_steps if total_steps > 0 else 0
                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR, 
                    f"tessera_game{game_num:05d}_loss{avg_loss:.4f}.pth"
                )
                save_checkpoint(model, optimizer, game_num, total_loss, avg_loss, ckpt_path)
                log(f"💾 Checkpoint saved: {ckpt_path}", log_file)
            
            # ゲーム終了後にデフラグ
            defragment_gpu_memory()
    
    except KeyboardInterrupt:
        log("\n⚠️ Training interrupted by user", log_file)
    
    except Exception as e:
        log(f"\n❌ Error: {str(e)}", log_file)
        log(traceback.format_exc(), log_file)
    
    finally:
        # 最終保存
        elapsed = time.time() - start_time
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        
        final_ckpt = os.path.join(
            config.CHECKPOINT_DIR,
            f"tessera_final_game{game_num}_loss{avg_loss:.4f}.pth"
        )
        save_checkpoint(model, optimizer, game_num, total_loss, avg_loss, final_ckpt)
        
        log("\n" + "=" * 60, log_file)
        log("📊 Training Summary", log_file)
        log("=" * 60, log_file)
        log(f"   Games completed: {game_num}", log_file)
        log(f"   Total time: {elapsed/60:.1f} minutes", log_file)
        log(f"   Average loss: {avg_loss:.4f}", log_file)
        log(f"   Final checkpoint: {final_ckpt}", log_file)
        
        monitor.print_summary()
        
        log("\n✅ Training complete!", log_file)


if __name__ == "__main__":
    main()
