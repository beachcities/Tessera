"""
Tessera Long Training Script v3.0
==================================
10時間以上の耐久運用向け全自動学習スクリプト

Design Philosophy:
- タイル化: 100ゲーム = 1タイル、タイル単位で管理
- 3層耐久性: コード / Docker / ホスト
- 文化継承: LLMが解析しやすいJSONLinesログ
- GPUネイティブ: メモリ効率最優先

Safety Features:
- 1ゲーム単位の例外保護
- SSM発散時の自動リセット
- OOM時の自動バッチサイズ縮小
- GPU温度監視（80℃で一時停止）
- NaN/Inf検知と自動復旧
- メモリリーク検知と定期GC
- チェックポイントからの自動再開

Scalability:
- RTX 4070 (8GB): BATCH_SIZE=16
- RTX 3090 (24GB): BATCH_SIZE=64
- A100 (80GB): BATCH_SIZE=256

Usage:
    python3.10 src/long_training_v3.py [--resume checkpoint.pth]

Version: 3.0.0
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import time
import datetime
import traceback
import gc
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, List, Dict, Any

# srcディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from monitor import TesseraMonitor, defragment_gpu_memory
from gpu_go_engine import GPUGoEngine, PASS_TOKEN, VOCAB_SIZE
from model import MambaModel, MambaStateCapture


# ============================================================
# Configuration with Auto-Scaling
# ============================================================

@dataclass
class ScalableConfig:
    """VRAM容量に応じて自動スケールする設定"""
    
    # === 基本設定 ===
    NUM_GAMES: int = 100000          # 総ゲーム数（10時間以上分）
    MOVES_PER_GAME: int = 150        # 1ゲームあたりの最大手数
    SEQ_LEN: int = 64                # シーケンス長
    
    # === タイル設定 ===
    TILE_SIZE: int = 100             # 1タイル = 100ゲーム
    
    # === モデル設定 ===
    D_MODEL: int = 256
    N_LAYERS: int = 4
    LEARNING_RATE: float = 1e-4
    WEIGHT_DECAY: float = 0.01
    
    # === ログ・保存設定 ===
    LOG_INTERVAL: int = 10           # ログ出力間隔（ゲーム数）
    CHECKPOINT_INTERVAL: int = 500   # チェックポイント間隔（ゲーム数）
    TILE_CHECKPOINT: bool = True     # タイルごとにもチェックポイント
    
    # === 安全装置の閾値 ===
    SSM_NORM_SOFT_LIMIT: float = 100.0   # 警告レベル
    SSM_NORM_HARD_LIMIT: float = 300.0   # リセットレベル
    MAX_VRAM_RATIO: float = 0.90         # VRAM使用率上限
    GRADIENT_CLIP_NORM: float = 0.5      # 勾配クリッピング
    
    # === GPU温度管理 ===
    GPU_TEMP_WARNING: int = 75           # 警告温度
    GPU_TEMP_THROTTLE: int = 80          # スロットル開始温度
    GPU_TEMP_PAUSE: int = 85             # 一時停止温度
    TEMP_CHECK_INTERVAL: int = 50        # 温度チェック間隔（ゲーム数）
    THROTTLE_SLEEP_SEC: float = 1.0      # スロットル時の待機時間
    
    # === メモリ管理 ===
    DEFRAG_INTERVAL: int = 50            # デフラグ間隔（手数）
    FULL_GC_INTERVAL: int = 100          # 完全GC間隔（ゲーム数）
    MEMORY_LEAK_THRESHOLD_MB: float = 500.0  # リーク検知閾値
    
    # === 自動リカバリ ===
    MAX_RETRIES_PER_GAME: int = 3        # 1ゲームあたりの最大リトライ
    MAX_CONSECUTIVE_FAILURES: int = 10   # 連続失敗の上限
    RESET_INTERVAL: int = 1000           # 定期リセット間隔（ゲーム数）
    
    # === 温度スケジューリング ===
    TEMP_START: float = 1.5              # 初期温度（探索重視）
    TEMP_END: float = 0.8                # 最終温度（収束重視）
    TEMP_DECAY_GAMES: int = 20000        # 温度減衰に要するゲーム数
    
    # === パス ===
    CHECKPOINT_DIR: str = "checkpoints"
    LOG_DIR: str = "logs"
    
    # === 動的に設定される値 ===
    BATCH_SIZE: int = 16
    
    @classmethod
    def auto_scale(cls, vram_gb: float) -> 'ScalableConfig':
        """VRAM容量に応じて自動スケール"""
        config = cls()
        
        if vram_gb >= 70:      # A100 80GB
            config.BATCH_SIZE = 256
            config.D_MODEL = 512
            config.N_LAYERS = 8
            config.TILE_SIZE = 200
        elif vram_gb >= 40:    # A100 40GB / A6000
            config.BATCH_SIZE = 128
            config.D_MODEL = 384
            config.N_LAYERS = 6
            config.TILE_SIZE = 150
        elif vram_gb >= 20:    # RTX 3090 / 4090
            config.BATCH_SIZE = 64
            config.D_MODEL = 256
            config.N_LAYERS = 4
        elif vram_gb >= 10:    # RTX 3080
            config.BATCH_SIZE = 32
            config.D_MODEL = 256
            config.N_LAYERS = 4
        else:                  # RTX 4070 Laptop (8GB)
            config.BATCH_SIZE = 64
            config.D_MODEL = 256
            config.N_LAYERS = 4
        
        return config


# ============================================================
# Tile Data Structure
# ============================================================

@dataclass
class TileMetadata:
    """タイルのメタデータ（LLM継承用）"""
    tile_id: int
    start_game: int
    end_game: int
    start_time: str
    end_time: str
    duration_sec: float
    games_played: int
    avg_loss: float
    min_loss: float
    max_loss: float
    total_moves: int
    resets: int
    errors: List[str] = field(default_factory=list)
    ssm_norm_avg: float = 0.0
    ssm_norm_max: float = 0.0
    vram_peak_mb: float = 0.0
    gpu_temp_max: int = 0
    status: str = "completed"


# ============================================================
# GPU Monitor
# ============================================================

class GPUMonitor:
    """GPU状態監視"""
    
    @staticmethod
    def get_temperature() -> int:
        """GPU温度を取得（℃）"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip().split('\n')[0])
        except:
            return 0  # 取得失敗時は0を返す
    
    @staticmethod
    def get_power_usage() -> float:
        """GPU消費電力を取得（W）"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip().split('\n')[0])
        except:
            return 0.0
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """GPUメモリ情報を取得"""
        if not torch.cuda.is_available():
            return {'used_mb': 0, 'total_mb': 0, 'free_mb': 0}
        
        return {
            'used_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'total_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }


# ============================================================
# Safety Manager
# ============================================================

class SafetyManager:
    """安全装置の管理"""
    
    def __init__(self, config: ScalableConfig):
        self.config = config
        self.reset_count = 0
        self.consecutive_failures = 0
        self.last_reset_game = 0
        self.oom_count = 0
        self.original_batch_size = config.BATCH_SIZE
        self.initial_vram_mb = 0.0
        self.errors: List[str] = []
        
    def set_initial_vram(self):
        """初期VRAM使用量を記録"""
        mem = GPUMonitor.get_memory_info()
        self.initial_vram_mb = mem.get('used_mb', 0)
    
    def check_memory_leak(self) -> Tuple[bool, float]:
        """メモリリークをチェック"""
        mem = GPUMonitor.get_memory_info()
        current = mem.get('used_mb', 0)
        increase = current - self.initial_vram_mb
        
        if increase > self.config.MEMORY_LEAK_THRESHOLD_MB:
            return True, increase
        return False, increase
    
    def check_gpu_temperature(self) -> Tuple[str, int]:
        """GPU温度をチェック"""
        temp = GPUMonitor.get_temperature()
        
        if temp >= self.config.GPU_TEMP_PAUSE:
            return "PAUSE", temp
        elif temp >= self.config.GPU_TEMP_THROTTLE:
            return "THROTTLE", temp
        elif temp >= self.config.GPU_TEMP_WARNING:
            return "WARNING", temp
        return "OK", temp
    
    def check_ssm_state(self, ssm_state: Optional[torch.Tensor]) -> Tuple[str, bool, float]:
        """SSM状態をチェック"""
        if ssm_state is None:
            return "OK", False, 0.0
        
        with torch.no_grad():
            norm = ssm_state.norm().item()
            
            if torch.isnan(ssm_state).any() or torch.isinf(ssm_state).any():
                return "NaN/Inf", True, norm
            
            if norm > self.config.SSM_NORM_HARD_LIMIT:
                return "DIVERGED", True, norm
            
            if norm > self.config.SSM_NORM_SOFT_LIMIT:
                return "WARNING", False, norm
        
        return "OK", False, norm
    
    def check_gradients(self, model: nn.Module) -> Tuple[str, bool]:
        """勾配をチェック"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return "NaN/Inf in gradients", True
                total_norm += p.grad.data.norm(2).item() ** 2
        
        total_norm = total_norm ** 0.5
        
        if total_norm > 100.0:
            return f"Gradient explosion ({total_norm:.1f})", True
        
        return "OK", False
    
    def check_loss(self, loss: float) -> Tuple[str, bool]:
        """Lossをチェック"""
        import math
        if math.isnan(loss) or math.isinf(loss):
            return "NaN/Inf loss", True
        if loss > 20.0:
            return f"Abnormal loss ({loss:.1f})", True
        return "OK", False
    
    def record_success(self):
        """成功を記録"""
        self.consecutive_failures = 0
    
    def record_failure(self, game_num: int, reason: str):
        """失敗を記録"""
        self.consecutive_failures += 1
        self.errors.append(f"Game {game_num}: {reason}")
        if len(self.errors) > 100:
            self.errors.pop(0)
    
    def record_reset(self, game_num: int, reason: str):
        """リセットを記録"""
        self.reset_count += 1
        self.last_reset_game = game_num
    
    def should_abort(self) -> Tuple[bool, str]:
        """完全停止が必要か判定"""
        if self.consecutive_failures >= self.config.MAX_CONSECUTIVE_FAILURES:
            return True, f"Too many consecutive failures ({self.consecutive_failures})"
        if self.oom_count >= 5:
            return True, f"Repeated OOM errors ({self.oom_count})"
        return False, ""
    
    def handle_oom(self) -> bool:
        """OOM発生時の処理。バッチサイズを縮小"""
        self.oom_count += 1
        
        if self.config.BATCH_SIZE > 4:
            self.config.BATCH_SIZE = self.config.BATCH_SIZE // 2
            return True
        return False
    
    def get_temperature(self, game_num: int) -> float:
        """現在の温度を計算（探索→収束）"""
        progress = min(1.0, game_num / self.config.TEMP_DECAY_GAMES)
        temp = self.config.TEMP_START - (self.config.TEMP_START - self.config.TEMP_END) * progress
        return temp
    
    def get_recent_errors(self, n: int = 5) -> List[str]:
        """直近のエラーを取得"""
        return self.errors[-n:]


# ============================================================
# Logger (JSON Lines + Console)
# ============================================================

class TesseraLogger:
    """デュアルログ（JSONL + コンソール）"""
    
    def __init__(self, log_dir: str):
        Path(log_dir).mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.console_log = os.path.join(log_dir, f"training_v3_{timestamp}.log")
        self.jsonl_log = os.path.join(log_dir, f"training_v3_{timestamp}.jsonl")
        self.tile_log = os.path.join(log_dir, f"tiles_v3_{timestamp}.jsonl")
    
    def log(self, message: str, also_print: bool = True):
        """コンソールログ"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        
        if also_print:
            print(line, flush=True)
        
        with open(self.console_log, "a") as f:
            f.write(line + "\n")
    
    def log_json(self, event_type: str, data: Dict[str, Any]):
        """JSONLログ（LLM継承用）"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": event_type,
            **data
        }
        with open(self.jsonl_log, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    def log_tile(self, tile: TileMetadata):
        """タイルメタデータをログ"""
        with open(self.tile_log, "a") as f:
            f.write(json.dumps(asdict(tile), ensure_ascii=False) + "\n")


# ============================================================
# Training Functions
# ============================================================

def save_checkpoint(model: nn.Module, 
                    optimizer: optim.Optimizer,
                    scheduler: Any,
                    game_num: int, 
                    stats: Dict,
                    config: ScalableConfig,
                    filepath: str):
    """完全なチェックポイント保存"""
    torch.save({
        'game': game_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'stats': stats,
        'config': asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }, filepath)


def load_checkpoint(filepath: str, 
                    model: nn.Module,
                    optimizer: optim.Optimizer,
                    scheduler: Any,
                    device: str) -> Tuple[int, Dict]:
    """チェックポイントから復元"""
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if checkpoint.get('rng_state') is not None:
        torch.set_rng_state(checkpoint['rng_state'])
    
    if checkpoint.get('cuda_rng_state') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    
    return checkpoint['game'], checkpoint.get('stats', {})


def reset_training_state(model: MambaModel, 
                         optimizer: optim.Optimizer,
                         config: ScalableConfig,
                         logger: TesseraLogger):
    """トレーニング状態をリセット（重みは保持）"""
    logger.log("🔄 Resetting training state (keeping weights)...")
    
    # Optimizerの状態をリセット
    optimizer.state.clear()
    
    # GPUメモリを完全クリア
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_single_game(engine: GPUGoEngine, 
                    model: MambaModel, 
                    state_capture: MambaStateCapture,
                    monitor: TesseraMonitor,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    config: ScalableConfig,
                    safety: SafetyManager,
                    device: str,
                    game_num: int) -> Tuple[float, int, str, float]:
    """
    1ゲームを実行して学習
    
    Returns:
        (loss, moves, status, ssm_norm)
    """
    engine.reset()
    state_capture.clear()
    
    game_moves = []
    temperature = safety.get_temperature(game_num)
    ssm_norm_max = 0.0
    
    for move_num in range(config.MOVES_PER_GAME):
        try:
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
                probs = model.get_move_probabilities(seq, legal_mask, temperature=temperature)
            
            if torch.isnan(probs).any():
                return 0.0, move_num, "NaN in probs", ssm_norm_max
            
            selected_moves = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # 着手
            engine.play_batch(selected_moves)
            game_moves.append(selected_moves.clone())
            
            # 終局チェック
            if engine.is_game_over().all():
                break
            
            # 定期的にSSM状態チェック
            if move_num % 20 == 0:
                ssm_state = state_capture.get_last_state()
                status, needs_reset, norm = safety.check_ssm_state(ssm_state)
                ssm_norm_max = max(ssm_norm_max, norm)
                
                if needs_reset:
                    return 0.0, move_num, status, ssm_norm_max
            
            # 定期的にデフラグ
            if move_num > 0 and move_num % config.DEFRAG_INTERVAL == 0:
                defragment_gpu_memory()
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return 0.0, move_num, "OOM", ssm_norm_max
            raise
    
    # 学習
    loss_value = 0.0
    if len(game_moves) > 1:
        try:
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
                
                logits = logits.contiguous()
                target_seq = target_seq.contiguous()
                
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))
                
                status, needs_reset = safety.check_loss(loss.item())
                if needs_reset:
                    return 0.0, len(game_moves), status, ssm_norm_max
                
                loss.backward()
                
                status, needs_reset = safety.check_gradients(model)
                if needs_reset:
                    optimizer.zero_grad()
                    return 0.0, len(game_moves), status, ssm_norm_max
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
                
                optimizer.step()
                
                loss_value = loss.item()
                
            # テンソルを明示的に削除
            del moves_tensor, input_seq, target_seq, logits
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                optimizer.zero_grad()
                defragment_gpu_memory()
                return 0.0, len(game_moves), "OOM", ssm_norm_max
            raise
    
    return loss_value, len(game_moves), "OK", ssm_norm_max


def main():
    """メインの学習ループ"""
    
    # コマンドライン引数
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # GPU情報取得とスケーリング
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        device = 'cuda'
    else:
        vram_gb = 0
        device = 'cpu'
    
    config = ScalableConfig.auto_scale(vram_gb)
    safety = SafetyManager(config)
    
    Path(config.CHECKPOINT_DIR).mkdir(exist_ok=True)
    logger = TesseraLogger(config.LOG_DIR)
    
    # ヘッダー
    logger.log("=" * 70)
    logger.log("🚀 Tessera Long Training v3.0 - 10-Hour Endurance Mode")
    logger.log("=" * 70)
    
    logger.log(f"Device: {device}")
    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        logger.log(f"GPU: {gpu_name} ({vram_gb:.1f} GB)")
        temp_status, temp = safety.check_gpu_temperature()
        logger.log(f"GPU Temperature: {temp}°C ({temp_status})")
    
    logger.log(f"\n📊 Auto-Scaled Config:")
    logger.log(f"   Batch size: {config.BATCH_SIZE}")
    logger.log(f"   Model: d={config.D_MODEL}, layers={config.N_LAYERS}")
    logger.log(f"   Games: {config.NUM_GAMES}")
    logger.log(f"   Tile size: {config.TILE_SIZE} games")
    logger.log(f"   Temperature: {config.TEMP_START} → {config.TEMP_END}")
    
    # 初期化
    logger.log("\n📦 Initializing components...")
    
    engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
    model = MambaModel(
        vocab_size=VOCAB_SIZE, 
        d_model=config.D_MODEL, 
        n_layers=config.N_LAYERS
    ).to(device)
    monitor = TesseraMonitor()
    state_capture = MambaStateCapture(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=362)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-6
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.log(f"Model parameters: {num_params:,}")
    
    # チェックポイントから再開
    start_game = 1
    stats = {
        'total_games': 0,
        'total_resets': 0,
        'best_loss': float('inf'),
        'total_moves': 0,
        'tiles_completed': 0,
    }
    
    if args.resume:
        logger.log(f"\n📂 Resuming from {args.resume}...")
        start_game, stats = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_game += 1
        logger.log(f"   Resumed at game {start_game}")
    
    # 初期VRAM記録
    safety.set_initial_vram()
    
    # JSONLログに開始を記録
    logger.log_json("training_start", {
        "config": asdict(config),
        "gpu": torch.cuda.get_device_name(0) if device == 'cuda' else "CPU",
        "vram_gb": vram_gb,
        "resume_from": args.resume,
    })
    
    # 学習ループ
    logger.log("\n🎮 Starting training loop...\n")
    
    start_time = time.time()
    total_loss = 0.0
    total_steps = 0
    losses_window = []
    game_num = start_game - 1
    
    # タイル管理
    current_tile_id = (start_game - 1) // config.TILE_SIZE
    tile_start_time = time.time()
    tile_start_game = start_game
    tile_losses = []
    tile_moves = 0
    tile_resets = 0
    tile_ssm_norms = []
    tile_errors = []
    vram_peak = 0.0
    gpu_temp_max = 0
    
    try:
        for game_num in range(start_game, config.NUM_GAMES + 1):
            
            # === タイル境界チェック ===
            new_tile_id = (game_num - 1) // config.TILE_SIZE
            if new_tile_id != current_tile_id and current_tile_id >= 0:
                # 前のタイルを完了
                tile_end_time = time.time()
                tile = TileMetadata(
                    tile_id=current_tile_id,
                    start_game=tile_start_game,
                    end_game=game_num - 1,
                    start_time=datetime.datetime.fromtimestamp(tile_start_time).isoformat(),
                    end_time=datetime.datetime.fromtimestamp(tile_end_time).isoformat(),
                    duration_sec=tile_end_time - tile_start_time,
                    games_played=game_num - tile_start_game,
                    avg_loss=sum(tile_losses) / len(tile_losses) if tile_losses else 0,
                    min_loss=min(tile_losses) if tile_losses else 0,
                    max_loss=max(tile_losses) if tile_losses else 0,
                    total_moves=tile_moves,
                    resets=tile_resets,
                    errors=tile_errors[-5:],  # 直近5エラー
                    ssm_norm_avg=sum(tile_ssm_norms) / len(tile_ssm_norms) if tile_ssm_norms else 0,
                    ssm_norm_max=max(tile_ssm_norms) if tile_ssm_norms else 0,
                    vram_peak_mb=vram_peak,
                    gpu_temp_max=gpu_temp_max,
                )
                logger.log_tile(tile)
                logger.log(f"📦 Tile {current_tile_id} completed: {tile.games_played} games, avg_loss={tile.avg_loss:.4f}")
                
                # タイルチェックポイント
                if config.TILE_CHECKPOINT:
                    ckpt_path = os.path.join(
                        config.CHECKPOINT_DIR,
                        f"tessera_tile{current_tile_id:05d}.pth"
                    )
                    save_checkpoint(model, optimizer, scheduler, game_num - 1, stats, config, ckpt_path)
                
                # 新タイル開始
                current_tile_id = new_tile_id
                tile_start_time = time.time()
                tile_start_game = game_num
                tile_losses = []
                tile_moves = 0
                tile_resets = 0
                tile_ssm_norms = []
                tile_errors = []
                vram_peak = 0.0
                gpu_temp_max = 0
            
            # === GPU温度チェック ===
            if game_num % config.TEMP_CHECK_INTERVAL == 0:
                temp_status, temp = safety.check_gpu_temperature()
                gpu_temp_max = max(gpu_temp_max, temp)
                
                if temp_status == "PAUSE":
                    logger.log(f"🌡️ GPU too hot ({temp}°C), pausing for 60s...")
                    time.sleep(60)
                elif temp_status == "THROTTLE":
                    logger.log(f"🌡️ GPU warm ({temp}°C), throttling...")
                    time.sleep(config.THROTTLE_SLEEP_SEC)
            
            # === メモリリークチェック ===
            if game_num % config.FULL_GC_INTERVAL == 0:
                has_leak, increase = safety.check_memory_leak()
                if has_leak:
                    logger.log(f"⚠️ Memory leak detected (+{increase:.0f}MB), forcing GC...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    safety.set_initial_vram()
                
                # VRAMピーク更新
                mem = GPUMonitor.get_memory_info()
                vram_peak = max(vram_peak, mem.get('used_mb', 0))
            
            # === 定期リセット ===
            if game_num > 1 and game_num % config.RESET_INTERVAL == 0:
                logger.log(f"🔄 Scheduled reset at game {game_num}")
                reset_training_state(model, optimizer, config, logger)
                engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
            
            # === ゲーム実行（リトライ付き） ===
            for retry in range(config.MAX_RETRIES_PER_GAME):
                try:
                    loss, moves, status, ssm_norm = run_single_game(
                        engine, model, state_capture, monitor, optimizer, criterion,
                        config, safety, device, game_num
                    )
                    break
                except Exception as e:
                    if retry < config.MAX_RETRIES_PER_GAME - 1:
                        logger.log(f"⚠️ Game {game_num} error (retry {retry+1}): {str(e)[:50]}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        loss, moves, status, ssm_norm = 0.0, 0, f"Exception: {str(e)[:50]}", 0.0
            
            # === 結果処理 ===
            tile_ssm_norms.append(ssm_norm)
            
            if status != "OK":
                safety.record_failure(game_num, status)
                tile_errors.append(status)
                tile_resets += 1
                
                if status == "OOM":
                    if not safety.handle_oom():
                        logger.log("❌ Cannot reduce batch size further")
                        break
                    logger.log(f"📉 Reduced batch size to {config.BATCH_SIZE}")
                    engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
                else:
                    reset_training_state(model, optimizer, config, logger)
                    engine = GPUGoEngine(batch_size=config.BATCH_SIZE, device=device)
                
                should_abort, reason = safety.should_abort()
                if should_abort:
                    logger.log(f"❌ Aborting: {reason}")
                    break
                
                continue
            
            # 成功
            safety.record_success()
            
            if loss > 0:
                total_loss += loss
                total_steps += 1
                losses_window.append(loss)
                tile_losses.append(loss)
                if len(losses_window) > 100:
                    losses_window.pop(0)
                
                if loss < stats['best_loss']:
                    stats['best_loss'] = loss
            
            tile_moves += moves
            stats['total_moves'] += moves
            
            scheduler.step()
            
            stats['total_games'] = game_num
            stats['total_resets'] = safety.reset_count
            
            # === 定期ログ ===
            if game_num % config.LOG_INTERVAL == 0:
                avg_loss = total_loss / total_steps if total_steps > 0 else 0
                recent_loss = sum(losses_window) / len(losses_window) if losses_window else 0
                elapsed = time.time() - start_time
                games_per_hour = game_num / (elapsed / 3600) if elapsed > 0 else 0
                eta_hours = (config.NUM_GAMES - game_num) / games_per_hour if games_per_hour > 0 else 0
                
                mem = GPUMonitor.get_memory_info()
                vram_pct = mem['used_mb'] / mem['total_mb'] * 100 if mem['total_mb'] > 0 else 0
                
                current_lr = optimizer.param_groups[0]['lr']
                temp = safety.get_temperature(game_num)
                
                logger.log(
                    f"Game {game_num:6d}/{config.NUM_GAMES} | "
                    f"Loss: {recent_loss:.4f} (best: {stats['best_loss']:.4f}) | "
                    f"VRAM: {vram_pct:.0f}% | "
                    f"Temp: {temp:.2f} | "
                    f"Resets: {safety.reset_count} | "
                    f"ETA: {eta_hours:.1f}h"
                )
                
                # JSONLログ
                logger.log_json("progress", {
                    "game": game_num,
                    "loss": recent_loss,
                    "best_loss": stats['best_loss'],
                    "vram_pct": vram_pct,
                    "resets": safety.reset_count,
                    "eta_hours": eta_hours,
                })
            
            # === チェックポイント保存 ===
            if game_num % config.CHECKPOINT_INTERVAL == 0:
                avg_loss = total_loss / total_steps if total_steps > 0 else 0
                ckpt_path = os.path.join(
                    config.CHECKPOINT_DIR, 
                    f"tessera_v3_game{game_num:06d}_loss{avg_loss:.4f}.pth"
                )
                save_checkpoint(model, optimizer, scheduler, game_num, stats, config, ckpt_path)
                logger.log(f"💾 Checkpoint: {ckpt_path}")
            
            # === 定期GC ===
            if game_num % 10 == 0:
                defragment_gpu_memory()
    
    except KeyboardInterrupt:
        logger.log("\n⚠️ Training interrupted by user")
    
    except Exception as e:
        logger.log(f"\n❌ Fatal error: {str(e)}")
        logger.log(traceback.format_exc())
    
    finally:
        # 最終タイルを記録
        if tile_losses:
            tile_end_time = time.time()
            tile = TileMetadata(
                tile_id=current_tile_id,
                start_game=tile_start_game,
                end_game=game_num,
                start_time=datetime.datetime.fromtimestamp(tile_start_time).isoformat(),
                end_time=datetime.datetime.fromtimestamp(tile_end_time).isoformat(),
                duration_sec=tile_end_time - tile_start_time,
                games_played=game_num - tile_start_game + 1,
                avg_loss=sum(tile_losses) / len(tile_losses),
                min_loss=min(tile_losses),
                max_loss=max(tile_losses),
                total_moves=tile_moves,
                resets=tile_resets,
                errors=tile_errors[-5:],
                ssm_norm_avg=sum(tile_ssm_norms) / len(tile_ssm_norms) if tile_ssm_norms else 0,
                ssm_norm_max=max(tile_ssm_norms) if tile_ssm_norms else 0,
                vram_peak_mb=vram_peak,
                gpu_temp_max=gpu_temp_max,
                status="interrupted" if game_num < config.NUM_GAMES else "completed"
            )
            logger.log_tile(tile)
        
        # 最終保存
        elapsed = time.time() - start_time
        avg_loss = total_loss / total_steps if total_steps > 0 else 0
        
        final_ckpt = os.path.join(
            config.CHECKPOINT_DIR,
            f"tessera_v3_final_game{game_num}_loss{avg_loss:.4f}.pth"
        )
        save_checkpoint(model, optimizer, scheduler, game_num, stats, config, final_ckpt)
        
        # サマリー
        logger.log("\n" + "=" * 70)
        logger.log("📊 Training Summary")
        logger.log("=" * 70)
        logger.log(f"   Games completed: {game_num}")
        logger.log(f"   Total time: {elapsed/60:.1f} min ({elapsed/3600:.2f} hours)")
        logger.log(f"   Average loss: {avg_loss:.4f}")
        logger.log(f"   Best loss: {stats['best_loss']:.4f}")
        logger.log(f"   Total moves: {stats['total_moves']:,}")
        logger.log(f"   Total resets: {safety.reset_count}")
        logger.log(f"   OOM events: {safety.oom_count}")
        logger.log(f"   Final batch size: {config.BATCH_SIZE}")
        logger.log(f"   Tiles completed: {current_tile_id + 1}")
        logger.log(f"   Final checkpoint: {final_ckpt}")
        
        # JSONLに終了を記録
        logger.log_json("training_end", {
            "games_completed": game_num,
            "total_hours": elapsed / 3600,
            "avg_loss": avg_loss,
            "best_loss": stats['best_loss'],
            "total_resets": safety.reset_count,
            "tiles_completed": current_tile_id + 1,
        })
        
        logger.log("\n✅ Training complete!")


if __name__ == "__main__":
    main()
