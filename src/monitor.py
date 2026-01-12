"""
TesseraMonitor - GPU Native 囲碁エンジンの観測系

設計思想:
- 観測自体がボトルネックにならない
- GPU同期は最小限（必要なときだけ）
- 異常検知と自動アラート
- Mamba SSM State の追跡

Version: 0.2.2
"""

import torch
import torch.nn.functional as F
import time
import gc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque


# ============================================================
# Data Classes
# ============================================================

@dataclass
class MambaStateMetrics:
    """Mamba SSM State の監視メトリクス"""
    ssm_state_size_mb: float = 0.0
    ssm_state_norm: float = 0.0
    ssm_state_entropy: float = 0.0
    conv_state_size_mb: float = 0.0
    ssm_state_delta_norm: float = 0.0
    ssm_state_max_activation: float = 0.0
    ssm_state_dead_units: int = 0
    ssm_state_saturated_units: int = 0


@dataclass
class FragmentationMetrics:
    """GPU メモリ断片化の監視メトリクス"""
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    fragmentation_ratio: float = 0.0
    num_alloc_retries: int = 0
    cached_mb: float = 0.0
    largest_free_block_mb: float = 0.0
    num_segments: int = 0


@dataclass
class Snapshot:
    """1時点の全観測データ"""
    timestamp: float
    move_number: int
    iteration: int = 0
    
    # Memory (基本)
    vram_used_mb: float = 0.0
    vram_reserved_mb: float = 0.0
    vram_max_mb: float = 0.0
    
    # Timing (ms)
    time_forward: float = 0.0
    time_legal_mask: float = 0.0
    time_play_move: float = 0.0
    time_transfer: float = 0.0
    time_ssm_step: float = 0.0
    
    # Game state
    stones_on_board: int = 0
    legal_moves_count: int = 0
    context_length: int = 0
    
    # Learning
    loss: float = 0.0
    gradient_norm: float = 0.0
    
    # Mamba State
    mamba: Optional[MambaStateMetrics] = None
    
    # Fragmentation
    fragmentation: Optional[FragmentationMetrics] = None


# ============================================================
# Main Monitor Class
# ============================================================

class TesseraMonitor:
    """
    Tessera/MambaGo 観測系
    
    最重要メトリクス Top 5:
    1. time_transfer - CPU-GPU転送時間（0であるべき）
    2. vram_used_mb - VRAM使用量
    3. fragmentation_ratio - メモリ断片化
    4. ssm_state_norm - SSM状態の発散検知
    5. move_number + legal_moves_count - ゲーム進行の健全性
    
    推奨ログ頻度:
    - time_forward, time_legal_mask, time_transfer: 毎手
    - ssm_state_norm, ssm_state_delta_norm: 10手ごと
    - fragmentation_ratio, vram_used_mb: 50手ごと
    - 全メトリクス詳細サマリー: 100手ごと
    """
    
    def __init__(self,
                 history_size: int = 1000,
                 alert_vram_threshold: float = 0.85,
                 alert_fragmentation_threshold: float = 1.5,
                 alert_state_norm_threshold: float = 100.0,
                 alert_transfer_threshold: float = 1.0):
        
        self.history: deque[Snapshot] = deque(maxlen=history_size)
        
        # アラート閾値
        self.alert_vram_threshold = alert_vram_threshold
        self.alert_fragmentation_threshold = alert_fragmentation_threshold
        self.alert_state_norm_threshold = alert_state_norm_threshold
        self.alert_transfer_threshold = alert_transfer_threshold
        
        # タイマー
        self._timers: Dict[str, float] = {}
        self._current_snapshot: Dict[str, Any] = {}
        
        # 前回のMamba State（差分計算用）
        self._prev_ssm_state: Optional[torch.Tensor] = None
        
        # 統計
        self._alert_count = 0

    # ============================================================
    # Memory Tracking
    # ============================================================
    
    def snapshot_vram(self) -> Dict[str, float]:
        """基本的なVRAM情報を取得"""
        if not torch.cuda.is_available():
            return {'used_mb': 0, 'reserved_mb': 0, 'max_mb': 0}
        
        return {
            'used_mb': torch.cuda.memory_allocated() / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
            'max_mb': torch.cuda.get_device_properties(0).total_memory / 1024**2,
        }
    
    def snapshot_fragmentation(self) -> FragmentationMetrics:
        """メモリ断片化の詳細情報を取得"""
        if not torch.cuda.is_available():
            return FragmentationMetrics()
        
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        
        # 断片化率: reserved が allocated より大きいほど断片化
        frag_ratio = reserved / allocated if allocated > 0 else 1.0
        
        # メモリ統計を取得
        try:
            stats = torch.cuda.memory_stats()
            num_alloc_retries = stats.get('num_alloc_retries', 0)
            num_segments = stats.get('num_segments', 0)
        except:
            num_alloc_retries = 0
            num_segments = 0
        
        return FragmentationMetrics(
            allocated_mb=allocated,
            reserved_mb=reserved,
            fragmentation_ratio=frag_ratio,
            num_alloc_retries=num_alloc_retries,
            num_segments=num_segments,
            largest_free_block_mb=reserved - allocated,
        )

    # ============================================================
    # Mamba State Tracking
    # ============================================================
    
    def snapshot_mamba_state(self,
                             ssm_state: Optional[torch.Tensor],
                             conv_state: Optional[torch.Tensor] = None) -> MambaStateMetrics:
        """
        Mamba の SSM State を分析
        
        Args:
            ssm_state: shape (batch, d_model, d_state) or similar
            conv_state: shape (batch, d_model, d_conv)
        """
        metrics = MambaStateMetrics()
        
        if ssm_state is None:
            return metrics
        
        with torch.no_grad():
            # サイズ
            metrics.ssm_state_size_mb = ssm_state.element_size() * ssm_state.numel() / 1024**2
            
            # ノルム（発散検知）
            metrics.ssm_state_norm = ssm_state.norm().item()
            
            # 最大活性値（飽和検知）
            metrics.ssm_state_max_activation = ssm_state.abs().max().item()
            
            # Dead units（活性がほぼゼロ）
            flat = ssm_state.view(-1)
            metrics.ssm_state_dead_units = (flat.abs() < 1e-6).sum().item()
            
            # Saturated units（上限に張り付き）
            metrics.ssm_state_saturated_units = (flat.abs() > 10.0).sum().item()
            
            # エントロピー（情報量の指標）
            try:
                probs = F.softmax(ssm_state.view(-1).float(), dim=0)
                metrics.ssm_state_entropy = -(probs * (probs + 1e-10).log()).sum().item()
            except:
                metrics.ssm_state_entropy = 0.0
            
            # 前ステップからの変化量
            if self._prev_ssm_state is not None:
                try:
                    if self._prev_ssm_state.shape == ssm_state.shape:
                        delta = ssm_state - self._prev_ssm_state
                        metrics.ssm_state_delta_norm = delta.norm().item()
                except:
                    pass
            
            # 現在の状態を保存
            self._prev_ssm_state = ssm_state.detach().clone()
        
        if conv_state is not None:
            metrics.conv_state_size_mb = conv_state.element_size() * conv_state.numel() / 1024**2
        
        return metrics

    # ============================================================
    # Timing
    # ============================================================
    
    def start_timer(self, name: str):
        """計測開始（GPU同期あり）"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._timers[name] = time.perf_counter()
    
    def stop_timer(self, name: str) -> float:
        """計測終了、ms で返す"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self._timers.get(name, time.perf_counter())) * 1000
        self._current_snapshot[f'time_{name}'] = elapsed
        return elapsed
    
    class _TimerContext:
        """タイマーのコンテキストマネージャー"""
        def __init__(self, monitor: 'TesseraMonitor', name: str):
            self.monitor = monitor
            self.name = name
        
        def __enter__(self):
            self.monitor.start_timer(self.name)
            return self
        
        def __exit__(self, *args):
            self.monitor.stop_timer(self.name)
    
    def time(self, name: str) -> _TimerContext:
        """
        コンテキストマネージャーでタイミング計測
        
        Usage:
            with monitor.time('forward'):
                output = model(input)
        """
        return self._TimerContext(self, name)

    # ============================================================
    # Recording
    # ============================================================
    
    def record(self,
               move_number: int,
               iteration: int = 0,
               ssm_state: Optional[torch.Tensor] = None,
               conv_state: Optional[torch.Tensor] = None,
               **kwargs) -> Snapshot:
        """
        スナップショットを記録
        
        Args:
            move_number: 現在の手数
            iteration: 学習イテレーション
            ssm_state: Mamba の SSM State（オプション）
            conv_state: Mamba の Conv State（オプション）
            **kwargs: 追加のメトリクス（stones_on_board, legal_moves_count 等）
        """
        vram = self.snapshot_vram()
        frag = self.snapshot_fragmentation()
        mamba = self.snapshot_mamba_state(ssm_state, conv_state)
        
        # タイミングデータをマージ
        timing_data = {k: v for k, v in self._current_snapshot.items() if k.startswith('time_')}
        
        snapshot = Snapshot(
            timestamp=time.time(),
            move_number=move_number,
            iteration=iteration,
            vram_used_mb=vram.get('used_mb', 0),
            vram_reserved_mb=vram.get('reserved_mb', 0),
            vram_max_mb=vram.get('max_mb', 0),
            mamba=mamba,
            fragmentation=frag,
            **timing_data,
            **kwargs
        )
        
        self.history.append(snapshot)
        self._check_alerts(snapshot)
        self._current_snapshot = {}
        
        return snapshot

    # ============================================================
    # Alerts
    # ============================================================
    
    def _check_alerts(self, s: Snapshot):
        """異常検知とアラート発報"""
        alerts = []
        
        # 1. CPU-GPU転送（最重要）
        if s.time_transfer > self.alert_transfer_threshold:
            alerts.append(f"🔴 TRANSFER {s.time_transfer:.1f}ms")
        
        # 2. VRAM使用率
        if s.vram_max_mb > 0:
            vram_ratio = s.vram_used_mb / s.vram_max_mb
            if vram_ratio > self.alert_vram_threshold:
                alerts.append(f"🔴 VRAM {vram_ratio*100:.1f}%")
        
        # 3. 断片化
        if s.fragmentation and s.fragmentation.fragmentation_ratio > self.alert_fragmentation_threshold:
            alerts.append(f"🟡 FRAG {s.fragmentation.fragmentation_ratio:.2f}x")
        
        # 4. SSM State 発散
        if s.mamba and s.mamba.ssm_state_norm > self.alert_state_norm_threshold:
            alerts.append(f"🔴 SSM_NORM {s.mamba.ssm_state_norm:.1f}")
        
        # 5. SSM State 停滞
        if s.mamba and s.move_number > 10 and s.mamba.ssm_state_delta_norm < 1e-6:
            alerts.append("🟡 SSM_STAGNANT")
        
        # 6. Dead units が多い
        if s.mamba and s.mamba.ssm_state_dead_units > 100:
            alerts.append(f"🟡 DEAD_UNITS {s.mamba.ssm_state_dead_units}")
        
        if alerts:
            self._alert_count += len(alerts)
            print(f"⚠️ Move {s.move_number}: {', '.join(alerts)}")

    # ============================================================
    # Analysis
    # ============================================================
    
    def memory_trend(self) -> Dict[str, List[tuple]]:
        """メモリ使用量の推移"""
        return {
            'vram': [(s.move_number, s.vram_used_mb) for s in self.history],
            'ssm_state': [(s.move_number, s.mamba.ssm_state_size_mb) 
                         for s in self.history if s.mamba],
            'fragmentation': [(s.move_number, s.fragmentation.fragmentation_ratio)
                             for s in self.history if s.fragmentation],
        }
    
    def mamba_state_trend(self) -> Dict[str, List[tuple]]:
        """Mamba State の推移"""
        return {
            'norm': [(s.move_number, s.mamba.ssm_state_norm) 
                    for s in self.history if s.mamba],
            'entropy': [(s.move_number, s.mamba.ssm_state_entropy)
                       for s in self.history if s.mamba],
            'delta': [(s.move_number, s.mamba.ssm_state_delta_norm)
                     for s in self.history if s.mamba],
            'dead_units': [(s.move_number, s.mamba.ssm_state_dead_units)
                          for s in self.history if s.mamba],
        }
    
    def find_anomalies(self) -> Dict[str, List[Snapshot]]:
        """各種異常を検出"""
        anomalies = {
            'memory_spike': [],
            'fragmentation_spike': [],
            'state_divergence': [],
            'state_stagnation': [],
        }
        
        prev = None
        for s in self.history:
            if prev:
                # メモリスパイク（50MB以上の急増）
                if s.vram_used_mb - prev.vram_used_mb > 50:
                    anomalies['memory_spike'].append(s)
                
                # 断片化スパイク
                if (s.fragmentation and prev.fragmentation and
                    s.fragmentation.fragmentation_ratio - prev.fragmentation.fragmentation_ratio > 0.2):
                    anomalies['fragmentation_spike'].append(s)
                
                # State発散
                if (s.mamba and prev.mamba and
                    s.mamba.ssm_state_norm > prev.mamba.ssm_state_norm * 1.5):
                    anomalies['state_divergence'].append(s)
                
                # State停滞
                if s.mamba and s.mamba.ssm_state_delta_norm < 1e-6 and s.move_number > 10:
                    anomalies['state_stagnation'].append(s)
            
            prev = s
        
        return anomalies

    # ============================================================
    # Output
    # ============================================================
    
    def print_summary(self):
        """詳細サマリーを表示"""
        print("\n" + "="*60)
        print("📊 Tessera Monitor Summary")
        print("="*60)
        
        if not self.history:
            print("No data recorded yet.")
            return
        
        latest = self.history[-1]
        
        # 基本情報
        print(f"\n🎮 Game State:")
        print(f"   Move: {latest.move_number}, Iteration: {latest.iteration}")
        print(f"   Stones: {latest.stones_on_board}, Legal moves: {latest.legal_moves_count}")
        
        # メモリ
        print(f"\n💾 Memory:")
        vram_pct = latest.vram_used_mb / latest.vram_max_mb * 100 if latest.vram_max_mb > 0 else 0
        print(f"   VRAM: {latest.vram_used_mb:.1f} / {latest.vram_max_mb:.1f} MB ({vram_pct:.1f}%)")
        if latest.fragmentation:
            f = latest.fragmentation
            print(f"   Fragmentation: {f.fragmentation_ratio:.2f}x (segments: {f.num_segments})")
        
        # Mamba State
        if latest.mamba:
            m = latest.mamba
            print(f"\n🐍 Mamba SSM State:")
            print(f"   Size: {m.ssm_state_size_mb:.2f} MB")
            print(f"   Norm: {m.ssm_state_norm:.2f}")
            print(f"   Entropy: {m.ssm_state_entropy:.2f}")
            print(f"   Delta: {m.ssm_state_delta_norm:.4f}")
            print(f"   Dead/Saturated: {m.ssm_state_dead_units}/{m.ssm_state_saturated_units}")
        
        # タイミング
        timing_attrs = ['time_forward', 'time_legal_mask', 'time_play_move', 'time_transfer']
        timing_values = [(attr, getattr(latest, attr, 0)) for attr in timing_attrs if getattr(latest, attr, 0) > 0]
        if timing_values:
            print(f"\n⏱️ Timing (latest):")
            for name, val in timing_values:
                print(f"   {name}: {val:.2f} ms")
        
        # トレンド分析
        if len(self.history) > 10:
            print(f"\n📈 Trends (first 10 → last 10):")
            
            early = list(self.history)[:10]
            late = list(self.history)[-10:]
            
            early_vram = sum(s.vram_used_mb for s in early) / 10
            late_vram = sum(s.vram_used_mb for s in late) / 10
            print(f"   VRAM: {early_vram:.1f} → {late_vram:.1f} MB ({late_vram-early_vram:+.1f})")
            
            early_mamba = [s for s in early if s.mamba]
            late_mamba = [s for s in late if s.mamba]
            if early_mamba and late_mamba:
                early_norm = sum(s.mamba.ssm_state_norm for s in early_mamba) / len(early_mamba)
                late_norm = sum(s.mamba.ssm_state_norm for s in late_mamba) / len(late_mamba)
                print(f"   SSM Norm: {early_norm:.2f} → {late_norm:.2f} ({late_norm-early_norm:+.2f})")
        
        # 異常サマリー
        anomalies = self.find_anomalies()
        total_anomalies = sum(len(v) for v in anomalies.values())
        print(f"\n⚠️ Alerts: {self._alert_count} total")
        if total_anomalies > 0:
            print(f"   Anomalies detected: {total_anomalies}")
            for name, items in anomalies.items():
                if items:
                    moves = [s.move_number for s in items[:5]]
                    suffix = '...' if len(items) > 5 else ''
                    print(f"   - {name}: moves {moves}{suffix}")
        
        print("\n" + "="*60)
    
    def get_top5_metrics(self) -> Dict[str, Any]:
        """最重要メトリクス Top 5 を取得"""
        if not self.history:
            return {}
        
        latest = self.history[-1]
        return {
            'time_transfer': latest.time_transfer,
            'vram_used_mb': latest.vram_used_mb,
            'fragmentation_ratio': latest.fragmentation.fragmentation_ratio if latest.fragmentation else 0,
            'ssm_state_norm': latest.mamba.ssm_state_norm if latest.mamba else 0,
            'legal_moves_count': latest.legal_moves_count,
        }


# ============================================================
# Utility Functions
# ============================================================

def defragment_gpu_memory():
    """
    GPUメモリの断片化を軽減
    学習の合間（例: 100手ごと）に呼ぶ
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.synchronize()


# ============================================================
# Mamba State Capture Hook
# ============================================================

class MambaStateCapture:
    """
    MambaBlock.forward に hook を挿入し、SSM State を取得する
    
    Usage:
        capture = MambaStateCapture(model)
        output = model(input)
        state = capture.get_state()
    """
    
    def __init__(self, model):
        self.captured_state = None
        self._hook_handle = None
        
        # Mamba モジュールを探してフックを登録
        self._register_hook(model)
    
    def _register_hook(self, model):
        """Mambaモジュールにフックを登録"""
        def hook_fn(module, input, output):
            # mamba-ssm の実装により異なる
            # 方法1: last_state 属性
            if hasattr(module, 'last_state'):
                self.captured_state = module.last_state.detach().clone()
            # 方法2: 出力がタプルの場合
            elif isinstance(output, tuple) and len(output) > 1:
                self.captured_state = output[1].detach().clone() if output[1] is not None else None
        
        # Mamba レイヤーを探す
        for name, module in model.named_modules():
            if 'mamba' in name.lower() or module.__class__.__name__ == 'Mamba':
                self._hook_handle = module.register_forward_hook(hook_fn)
                break
    
    def get_state(self) -> Optional[torch.Tensor]:
        """キャプチャした状態を取得"""
        return self.captured_state
    
    def remove_hook(self):
        """フックを解除"""
        if self._hook_handle:
            self._hook_handle.remove()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing TesseraMonitor...")
    
    # モニター初期化
    monitor = TesseraMonitor()
    
    # ダミーデータで記録テスト
    for move in range(20):
        with monitor.time('forward'):
            time.sleep(0.001)  # 1ms のダミー処理
        
        with monitor.time('legal_mask'):
            time.sleep(0.0005)
        
        # ダミーの SSM State
        if torch.cuda.is_available():
            dummy_state = torch.randn(4, 256, 16, device='cuda')
        else:
            dummy_state = torch.randn(4, 256, 16)
        
        monitor.record(
            move_number=move,
            ssm_state=dummy_state,
            stones_on_board=move * 2,
            legal_moves_count=361 - move * 2,
        )
    
    # サマリー表示
    monitor.print_summary()
    
    # Top 5 メトリクス
    print("\nTop 5 Metrics:")
    print(monitor.get_top5_metrics())
    
    print("\n✅ TesseraMonitor test passed!")
