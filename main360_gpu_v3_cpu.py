import time
import sys
import random
import ray
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
from collections import deque
from torch.cuda.amp import autocast, GradScaler

# --- Config ---
VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND = 100, 256, 16, 4, 2

# 【V3変更点】ActorをCPUへ完全退避
# RTX 4070 (8GB) のVRAMをLearnerに集中させるための設定
NUM_ACTORS = 8 
ACTOR_GPU_FRACTION = 0   # 0 = CPUのみ使用
LEARNER_GPU_FRACTION = 0.9 # GPUの大半をLearnerへ

ACTOR_BATCH, SEQ_LEN = 4, 32
REPLAY_CAPACITY, BACKPRESSURE_THRESHOLD = 5000, 4000
LOGICAL_BATCH_SIZE, MICRO_BATCH_SIZE = 32, 8
ACCUM_STEPS = LOGICAL_BATCH_SIZE // MICRO_BATCH_SIZE
LEARNER_MAX_STEPS = 5000 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Models ---
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

@ray.remote
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    def add(self, sample): self.buffer.append(sample)
    def size(self): return len(self.buffer)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size) if len(self.buffer) >= batch_size else []
    def get_stats(self): return len(self.buffer)

@ray.remote
class ParameterServer:
    def __init__(self, model_state=None): 
        self.state = model_state
        self.update_count = 0
    def set_weights(self, state_dict): 
        self.state = {k: v.cpu() for k, v in state_dict.items()}
        self.update_count += 1
        return self.update_count
    def get_weights(self): return self.state
    def get_version(self): return self.update_count

@ray.remote(num_gpus=LEARNER_GPU_FRACTION if torch.cuda.is_available() else 0)
class Learner:
    def __init__(self, replay_actor, param_server):
        try:
            self.device = torch.device(DEVICE)
            self.replay, self.param_server = replay_actor, param_server
            print(f"🧠 Learner: Init on {self.device} (AMP+Accumulation)")
            self.model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND).to(self.device)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
            self.scaler = GradScaler()
            
            state = ray.get(self.param_server.get_weights.remote())
            if state: self.model.load_state_dict(state)
        except Exception as e:
            print(f"❌ Learner Init Error: {e}")
            traceback.print_exc()
            raise e

    def train_loop(self, max_steps):
        try:
            step = 0
            loss_history = []
            print("🧠 Learner: Starting Training Loop...")
            
            while step < max_steps:
                while ray.get(self.replay.size.remote()) < MICRO_BATCH_SIZE:
                    time.sleep(0.01)

                batch = ray.get(self.replay.sample.remote(LOGICAL_BATCH_SIZE))
                if not batch: continue
                
                inputs_full = torch.stack([b[0] for b in batch]).to(self.device, non_blocking=True)
                targets_full = torch.stack([b[1] for b in batch]).to(self.device, non_blocking=True)
                
                self.model.train()
                self.optimizer.zero_grad()
                
                total_loss = 0
                for i in range(0, LOGICAL_BATCH_SIZE, MICRO_BATCH_SIZE):
                    with autocast():
                        logits = self.model(inputs_full[i:i+MICRO_BATCH_SIZE])
                        loss = self.criterion(logits.reshape(-1, VOCAB_SIZE), targets_full[i:i+MICRO_BATCH_SIZE].view(-1)) / ACCUM_STEPS
                    self.scaler.scale(loss).backward()
                    total_loss += loss.item()
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                step += 1
                loss_history.append(total_loss * ACCUM_STEPS)
                if len(loss_history) > 100: loss_history.pop(0)

                if step % 10 == 0: 
                    self.param_server.set_weights.remote({k: v.cpu() for k, v in self.model.state_dict().items()})
                
                if step % 50 == 0:
                    avg_loss = sum(loss_history) / len(loss_history)
                    print(f"🧠 Learner Step {step:04d} | Loss: {avg_loss:.4f}")
                    
            return "done"
        except Exception as e:
            print(f"❌ Learner Crash at step {step}: {e}")
            traceback.print_exc()
            raise e

@ray.remote(num_gpus=ACTOR_GPU_FRACTION if torch.cuda.is_available() else 0)
class SelfPlayActor:
    def __init__(self, actor_id, replay_actor, param_server):
        self.actor_id = actor_id
        # ActorはCPU強制
        self.device = torch.device("cpu") 
        self.replay = replay_actor
        self.param_server = param_server
        self.local_version = -1
        
        print(f"🤖 Actor {actor_id}: Init on {self.device}")
        self.model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND).to(self.device)
        
        with torch.inference_mode(): 
            self.model(torch.randint(0, VOCAB_SIZE, (ACTOR_BATCH, SEQ_LEN)).to(self.device))

    def step(self):
        try:
            if ray.get(self.replay.size.remote(), timeout=0.01) > BACKPRESSURE_THRESHOLD:
                return "wait"
        except: pass

        if random.random() < 0.05: 
            latest_ver = ray.get(self.param_server.get_version.remote())
            if latest_ver > self.local_version:
                state = ray.get(self.param_server.get_weights.remote())
                self.model.load_state_dict(state)
                self.local_version = latest_ver

        with torch.inference_mode():
            inp = torch.randint(0, VOCAB_SIZE, (ACTOR_BATCH, SEQ_LEN)).to(self.device)
            tgt = torch.randint(0, VOCAB_SIZE, (ACTOR_BATCH, SEQ_LEN)).to(self.device)
            samples = [(inp.cpu()[i], tgt.cpu()[i]) for i in range(ACTOR_BATCH)]
            
        self.replay.add.remote(samples[0])
        return "gen"

def start():
    if ray.is_initialized(): ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False, num_gpus=1)
    
    print("🚀 Project initialized. Starting Actors (CPU) & Learner (GPU)...")

    replay = ReplayBuffer.remote(REPLAY_CAPACITY)
    init_model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND)
    param = ParameterServer.remote({k: v.cpu() for k, v in init_model.state_dict().items()})
    
    learner = Learner.remote(replay, param)
    learner_handle = learner.train_loop.remote(max_steps=LEARNER_MAX_STEPS)
    
    actors = []
    for i in range(NUM_ACTORS):
        actors.append(SelfPlayActor.remote(i, replay, param))
        time.sleep(0.1) 

    pending_tasks = {a.step.remote(): a for a in actors}
    
    start_time = time.time()
    steps = 0
    
    try:
        while len(pending_tasks) > 0:
            done_ids, _ = ray.wait(list(pending_tasks.keys()), num_returns=1)
            
            for done_id in done_ids:
                actor = pending_tasks.pop(done_id)
                steps += 1
                new_task = actor.step.remote()
                pending_tasks[new_task] = actor
            
            ready_learner, _ = ray.wait([learner_handle], timeout=0)
            if ready_learner:
                try:
                    res = ray.get(learner_handle)
                    print(f"\n✅ Learner finished training: {res}")
                except Exception as e:
                    print(f"\n❌ Learner CRASHED during training!")
                    print(e)
                break

            if steps % 100 == 0:
                elapsed = time.time() - start_time
                fps = steps / elapsed
                replay_size = ray.get(replay.get_stats.remote())
                sys.stdout.write(f"\rStep {steps} | FPS: {fps:.1f} | Replay: {replay_size}/{REPLAY_CAPACITY}")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    start()