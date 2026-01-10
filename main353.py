import os
import sys
import torch
import shutil
import time
import traceback
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.amp import autocast
from multiprocessing import cpu_count

# --- ライブラリチェック ---
print("🔍 DEBUG: Checking libraries...")
try:
    import mamba_ssm
    import causal_conv1d
    print(f"✅ Mamba Version: {mamba_ssm.__version__}")
except ImportError as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

try:
    import ray
except ImportError:
    print("⚠️ Ray is missing. Installing...")
    os.system("pip install ray[default]")
    import ray

# ==========================================
# 【Block 1】 環境設定
# ==========================================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
LOCAL_CKPT = os.path.join(BASE_DIR, "latest_model.pth")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# VRAMの断片化を防ぐ設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Environment: Local Docker / Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ==========================================
# 【Block 2】 モデル定義
# ==========================================
class GoMambaSpatialNet(nn.Module):
    def __init__(self, d_model=512, n_layers=12, d_state=16):
        super().__init__()
        self.input_embed = nn.Sequential(
            nn.Conv2d(17, d_model // 4, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model // 4, d_model, kernel_size=1)
        )
        self.spatial_link = nn.Parameter(torch.randn(361, d_model) * 0.02)
        self.layers = nn.ModuleList([
            mamba_ssm.Mamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, 362)
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, x):
        device, dtype = x.device, x.dtype
        x = self.input_embed(x).flatten(2).transpose(1, 2)
        x = x + self.spatial_link.to(device=device, dtype=dtype)
        for layer in self.layers:
            x = x + layer(x)
        x = self.norm_f(x)
        x_glob = x.mean(dim=1)
        return self.policy_head(x_glob), torch.tanh(self.value_head(x_glob))

# ==========================================
# 【Block 3】 物理エンジン
# ==========================================
class TorchGo:
    def __init__(self, batch_size=128, device='cuda'):
        self.B, self.device = batch_size, device
        self.states = torch.zeros(self.B, 17, 19, 19, device=device)
        self.move_counts = torch.zeros(self.B, device=device, dtype=torch.long)

    def reset(self, indices=None):
        if indices is None: 
            self.states.zero_()
            self.move_counts.zero_()
        else: 
            self.states[indices] = 0
            self.move_counts[indices] = 0

    def step(self, actions):
        self.move_counts += 1
        dones = self.move_counts >= 150
        rewards = torch.zeros(self.B, device=self.device)
        if dones.any():
            rewards[dones] = torch.randn(dones.sum().item(), device=self.device).sign()
        return self.states, rewards, dones

def get_target_p_from_mcts(visit_counts, temp=1.0):
    p = visit_counts / (visit_counts.sum(dim=-1, keepdim=True) + 1e-8)
    return p

# ==========================================
# 【Block 4】 分散Actor (GPU版)
# ==========================================
if ray.is_initialized(): ray.shutdown()
# GPUリソースを認識させる
ray.init(ignore_reinit_error=True, num_gpus=1)

# ⚠️ num_gpus=0.05 に設定
# これにより、1枚のGPUに最大20個のActorが同居できるようになります。
@ray.remote(num_gpus=0.05) 
class SelfPlayActorV3:
    def __init__(self, sd_or_ref):
        print("🤖 Actor: Initializing on GPU...")
        # ActorもGPUを使う
        self.device = 'cuda'
        self.dtype = torch.bfloat16 # RTX 4070はbf16が得意
        
        if isinstance(sd_or_ref, dict):
            sd_dict = sd_or_ref
        else:
            sd_dict = ray.get(sd_or_ref)

        self.model = GoMambaSpatialNet().to(self.device).to(self.dtype)
        # GPUへ転送
        gpu_sd = {k: v.to(self.device).to(self.dtype) for k, v in sd_dict.items()}
        self.model.load_state_dict(gpu_sd)
        
        # バッチサイズ64ならVRAM消費は軽微
        self.env = TorchGo(batch_size=64, device=self.device)
        print("🤖 Actor: Ready.")

    def play_and_report(self, sd_or_ref=None):
        if sd_or_ref is not None:
            if isinstance(sd_or_ref, dict):
                sd_dict = sd_or_ref
            else:
                sd_dict = ray.get(sd_or_ref)
            gpu_sd = {k: v.to(self.device).to(self.dtype) for k, v in sd_dict.items()}
            self.model.load_state_dict(gpu_sd)

        self.model.eval()
        self.env.reset()
        
        # GPU上での高速推論
        for _ in range(50):
            with torch.no_grad(), autocast(device_type='cuda', dtype=torch.bfloat16):
                p_out, _ = self.model(self.env.states)
                actions = torch.multinomial(F.softmax(p_out, dim=-1), 1).squeeze()
                self.env.step(actions)

        # 結果を返すときはCPUに戻す（GPUメモリ解放 & 通信量削減）
        visit_counts = torch.randn(self.env.B, 362, device='cuda').abs()
        target_p = get_target_p_from_mcts(visit_counts)
        
        return (self.env.states.cpu(), target_p.cpu(), torch.randn(self.env.B).cpu())

# ==========================================
# 【Block 5】 学習ループ
# ==========================================
def start_training():
    print(f"🏁 DEBUG: start_training() called.")
    device = DEVICE
    
    print("🔧 DEBUG: Initializing Model on GPU...")
    model = GoMambaSpatialNet().to(device)
    ema_model = deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    print("✅ DEBUG: Model initialized.")

    # 重みはCPU経由で共有（Rayの作法として安全）
    current_sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    sd_ref = ray.put(current_sd_cpu)

    # Actor数調整: VRAM 12GB なら 8〜12個くらいが安全圏
    num_actors = 8 
    print(f"👥 DEBUG: Spawning {num_actors} GPU Actors via Ray...")
    
    actors = [SelfPlayActorV3.remote(sd_ref) for _ in range(num_actors)]
    print("✅ DEBUG: Actors spawned. Entering loop...")

    t_start = time.time()

    for step in range(1, 10001):
        if step == 1: print("🔄 DEBUG: Step 1 started.")

        futures = [a.play_and_report.remote(sd_ref) for a in actors]
        ready, remaining = ray.wait(futures, num_returns=len(futures), timeout=10)

        if len(ready) == 0:
            if step == 1: print("⏳ DEBUG: Waiting for actors (Timeout)...")
            time.sleep(0.1)
            continue

        results = ray.get(ready)

        s_list = [r[0] for r in results]
        p_list = [r[1] for r in results]
        v_list = [r[2] for r in results]

        states_all = torch.cat(s_list, dim=0).to(device, non_blocking=True)
        tp_all = torch.cat(p_list, dim=0).to(device, non_blocking=True)
        tv_all = torch.cat(v_list, dim=0).to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            p_out, v_out = model(states_all)
            loss = -torch.mean(torch.sum(tp_all * F.log_softmax(p_out, dim=-1), dim=-1)) + \
                   0.5 * F.mse_loss(v_out.squeeze(), tv_all.to(torch.bfloat16))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            for e_p, p in zip(ema_model.parameters(), model.parameters()):
                e_p.data.mul_(0.99).add_(p.data, alpha=0.01)

        if step % 10 == 0:
            current_sd_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            sd_ref = ray.put(current_sd_cpu)

            elapsed = time.time() - t_start
            res_mem = torch.cuda.memory_reserved(device) / 1e9 if torch.cuda.is_available() else 0
            print(f"Iter {step:04} | Loss: {loss.item():.4f} | Batch: {len(states_all)} | VRAM: {res_mem:.2f}GB")

            if step % 50 == 0:
                torch.save(ema_model.state_dict(), LOCAL_CKPT)

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        print("🚀 Main: Starting program...")
        start_training()
        print("✅ Main: Program finished successfully.")
    except Exception:
        print("\n🚨 CRITICAL ERROR OCCURRED! (詳細なエラーログ)")
        print("="*60)
        traceback.print_exc()
        print("="*60)
    except KeyboardInterrupt:
        print("\n🛑 Training Stopped by User.")
    finally:
        if ray.is_initialized():
            ray.shutdown()
            print("INFO: Ray shutdown complete.")