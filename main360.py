import time
import sys
import random
import ray
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.cuda.amp import autocast, GradScaler

# -------------------------
# Config: RTX 4070 (8GB) Optimized
# -------------------------
VOCAB_SIZE = 100
D_MODEL = 256
D_STATE = 16
D_CONV = 4
EXPAND = 2

NUM_ACTORS = 8
# GPU配分: Learner 0.5 + Actors 0.4 (0.05*8) = 0.9 < 1.0
ACTOR_GPU_FRACTION = 0.05  
LEARNER_GPU_FRACTION = 0.5 

ACTOR_BATCH = 4
SEQ_LEN = 32

REPLAY_CAPACITY = 5000
# Backpressure: Replayがこのラインを超えたらActorは休憩する
BACKPRESSURE_THRESHOLD = 4000 

# 学習設定 (VRAM節約のための勾配累積)
LOGICAL_BATCH_SIZE = 32   # 学習として進めたいバッチサイズ
MICRO_BATCH_SIZE = 8      # VRAMに乗せる物理バッチサイズ
ACCUM_STEPS = LOGICAL_BATCH_SIZE // MICRO_BATCH_SIZE

LEARNER_MAX_STEPS = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Model
# -------------------------
class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        from mamba_ssm import Mamba
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba(x)
        return self.head(x)

# -------------------------
# ReplayBuffer Actor
# -------------------------
@ray.remote
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, sample):
        # sample: (input_cpu, target_cpu)
        self.buffer.append(sample)
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        batch = random.sample(self.buffer, batch_size)
        return batch

# -------------------------
# Parameter Server Actor
# -------------------------
@ray.remote
class ParameterServer:
    def __init__(self, model_state=None):
        self.state = model_state

    def set_weights(self, state_dict):
        # メモリ節約のためCPUで保持
        self.state = {k: v.cpu() for k, v in state_dict.items()}
        return True

    def get_weights(self):
        return self.state

# -------------------------
# Learner Actor (AMP + Gradient Accumulation)
# -------------------------
@ray.remote(num_gpus=LEARNER_GPU_FRACTION if torch.cuda.is_available() else 0)
class Learner:
    def __init__(self, replay_actor, param_server):
        self.device = torch.device(DEVICE)
        self.replay = replay_actor
        self.param_server = param_server

        print(f"🧠 Learner: Initializing on {self.device} (AMP Enabled)...")
        self.model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        
        # AMP用スケーラー
        self.scaler = GradScaler()

        # 初期重み同期
        state = ray.get(self.param_server.get_weights.remote())
        if state is not None:
            self.model.load_state_dict(state)

    def train_loop(self, max_steps):
        step = 0
        last_log_time = time.time()
        
        while step < max_steps:
            # 1. データサンプリング
            batch = ray.get(self.replay.sample.remote(LOGICAL_BATCH_SIZE))
            if not batch:
                time.sleep(0.01)
                continue

            # 2. データをGPUへ
            inputs_full = torch.stack([b[0] for b in batch]).to(self.device)
            targets_full = torch.stack([b[1] for b in batch]).to(self.device)

            self.model.train()
            self.optimizer.zero_grad()

            # 3. 勾配累積ループ (Micro-Batching)
            total_loss = 0
            for i in range(0, LOGICAL_BATCH_SIZE, MICRO_BATCH_SIZE):
                # VRAMに優しいサイズに切り出す
                inputs = inputs_full[i : i + MICRO_BATCH_SIZE]
                targets = targets_full[i : i + MICRO_BATCH_SIZE]

                # AMP Context (混合精度計算)
                with autocast():
                    logits = self.model(inputs)
                    loss = self.criterion(logits.reshape(-1, VOCAB_SIZE), targets.view(-1))
                    # 累積用にLossを割っておく
                    loss = loss / ACCUM_STEPS
                
                # Scaled Backward
                self.scaler.scale(loss).backward()
                total_loss += loss.item()

            # 4. Update Weights
            self.scaler.step(self.optimizer)
            self.scaler.update()

            step += 1

            # 5. 重み公開 (少し頻度を下げる)
            if step % 10 == 0:
                state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                self.param_server.set_weights.remote(state)

            # 6. ログ出力
            if step % 20 == 0:
                elapsed = time.time() - last_log_time
                steps_per_sec = 20 / elapsed
                replay_size = ray.get(self.replay.size.remote())
                print(f"🧠 Learner Step {step:04d} | Loss: {total_loss * ACCUM_STEPS:.4f} | Speed: {steps_per_sec:.1f} step/s | Replay: {replay_size}")
                last_log_time = time.time()

        return "learner_done"

# -------------------------
# Actor (With Backpressure)
# -------------------------
@ray.remote(num_gpus=ACTOR_GPU_FRACTION if torch.cuda.is_available() else 0)
class SelfPlayActor:
    def __init__(self, actor_id, replay_actor, param_server):
        self.actor_id = actor_id
        self.device = torch.device(DEVICE)
        self.replay = replay_actor
        self.param_server = param_server

        print(f"🤖 Actor {actor_id}: Init on {self.device}")
        self.model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND).to(self.device)
        
        # Warm-up
        dummy = torch.randint(0, VOCAB_SIZE, (ACTOR_BATCH, SEQ_LEN)).to(self.device)
        with torch.inference_mode():
            self.model(dummy)

    def generate_and_send(self):
        # ★ Backpressure Implementation ★
        # ReplayBufferが溢れそうなら、データ生成せずに待機する
        try:
            current_size = ray.get(self.replay.size.remote(), timeout=0.1)
            if current_size > BACKPRESSURE_THRESHOLD:
                # 詰まっている時は少し休む (CPU/Network負荷軽減)
                time.sleep(0.1) 
                return "backpressure_wait"
        except:
            pass # タイムアウト時は強行突破（停止させないため）

        # Inference
        with torch.inference_mode():
            inp = torch.randint(0, VOCAB_SIZE, (ACTOR_BATCH, SEQ_LEN)).to(self.device)
            tgt = torch.randint(0, VOCAB_SIZE, (ACTOR_BATCH, SEQ_LEN)).to(self.device)
            
            # CPU退避 (重要)
            inp_cpu = inp.cpu()
            tgt_cpu = tgt.cpu()
            
            samples = [(inp_cpu[i], tgt_cpu[i]) for i in range(ACTOR_BATCH)]

        # Push
        for s in samples:
            self.replay.add.remote(s)
            
        return "generated"

    def sync_weights(self):
        state = ray.get(self.param_server.get_weights.remote())
        if state is not None:
            self.model.load_state_dict(state)

# -------------------------
# Main Execution
# -------------------------
def start():
    print("🚀 Main: Starting GPU Full-Stack RL (AMP + Accumulation)...")
    
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    replay = ReplayBuffer.remote(REPLAY_CAPACITY)
    init_model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND)
    param_server = ParameterServer.remote({k: v.cpu() for k, v in init_model.state_dict().items()})

    # Learner起動 (GPU)
    learner = Learner.remote(replay, param_server)
    learner_future = learner.train_loop.remote(max_steps=LEARNER_MAX_STEPS)

    # Actors起動 (GPU)
    actors = [SelfPlayActor.remote(i, replay, param_server) for i in range(NUM_ACTORS)]

    try:
        start_time = time.time()
        step = 0
        
        while True:
            # Actors駆動
            results = ray.get([actor.generate_and_send.remote() for actor in actors])
            
            # 定期同期
            if step % 200 == 0:
                [actor.sync_weights.remote() for actor in actors]

            # Learner監視 (ノンブロッキング)
            done, _ = ray.wait([learner_future], timeout=0)
            if done:
                print("\n✅ Learner finished training!")
                break
                
            # メインループの進捗表示
            if step % 50 == 0:
                wait_count = results.count("backpressure_wait")
                replay_size = ray.get(replay.size.remote())
                sys.stdout.write(f"\r⏱️  Main Step: {step} | Replay: {replay_size} | Backpressure: {wait_count} actors waiting")
                sys.stdout.flush()

            step += 1
            # Actorsが早すぎる場合の調整 (Learner GPUの負荷を考慮)
            time.sleep(0.02) 

    except KeyboardInterrupt:
        print("\n🛑 Stop.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    start()

