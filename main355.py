import os
import time
import sys
import torch
import torch.nn as nn
import ray
from mamba_ssm import Mamba

# ---------------------------------------------------------
# 1. 設定 (Configuration)
# ---------------------------------------------------------
VOCAB_SIZE = 100
D_MODEL = 256
D_STATE = 16
D_CONV = 4
EXPAND = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ★ いよいよ本番構成：8並列 ★
NUM_ACTORS = 8     
BATCH_SIZE = 4
SEQ_LEN = 32

# ---------------------------------------------------------
# 2. モデル定義
# ---------------------------------------------------------
class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_state, d_conv, expand):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.mamba(x)
        return self.head(x)

# ---------------------------------------------------------
# 3. Ray Actor定義 (Warm-up & Inference Mode搭載)
# ---------------------------------------------------------
@ray.remote(num_gpus=0.1)
class SelfPlayActorV3:
    def __init__(self, actor_id):
        self.actor_id = actor_id
        self.device = DEVICE
        
        # モデル初期化
        self.model = MambaModel(VOCAB_SIZE, D_MODEL, D_STATE, D_CONV, EXPAND).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        
        # データ準備
        self.dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(self.device)
        self.dummy_target = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(self.device)

        # ★ Warm-up: 初回コンパイルをここで済ませる
        print(f"(Actor {actor_id}) 🔨 Warming up (Compiling Mamba)...")
        self._run_forward_pass() 
        print(f"(Actor {actor_id}) ✅ Ready (Warm-up complete).")

    def _run_forward_pass(self):
        # 共通処理として切り出し
        # 今回は学習シミュレーションだが、メモリ節約のため勾配計算なしモードで動かす
        with torch.inference_mode():
            logits = self.model(self.dummy_input)
            # ★ 安全な reshape に変更
            logits_flat = logits.reshape(-1, VOCAB_SIZE)
            target_flat = self.dummy_target.view(-1)
            loss = self.criterion(logits_flat, target_flat)
        return loss.item()

    def train_step(self):
        # 本番ループではこれを呼ぶだけ（すでにコンパイル済みで高速）
        loss_val = self._run_forward_pass()
        return {"loss": loss_val, "id": self.actor_id}

# ---------------------------------------------------------
# 4. メイン実行ループ
# ---------------------------------------------------------
def start_training():
    print(f"🚀 Main: Starting program ({NUM_ACTORS} Actors Mode) on {DEVICE}...")

    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, log_to_driver=False)

    print(f"👥 DEBUG: Spawning {NUM_ACTORS} GPU Actors (This will take time for Warm-up)...")
    
    # ここで各Actorの __init__ が走り、コンパイル待ちが発生する
    # しかし、メインループに入る前に終わるので UX が良い
    actors = [SelfPlayActorV3.remote(i) for i in range(NUM_ACTORS)]
    
    # 全員が Ready になるまで待つ手もあるが、Rayは遅延実行してくれるのでそのまま進む
    print("✅ DEBUG: Actors spawned. Starting loop...")
    print("-" * 60)

    try:
        for step in range(1, 1001):
            
            # --- 並列実行 ---
            futures = [actor.train_step.remote() for actor in actors]
            
            start_wait = time.time()
            pending = futures
            
            # Warm-up済みなので、Step1から高速なはず！
            guideline_limit = 30 # もう300秒も待つ必要はない
            
            while len(pending) > 0:
                finished, pending = ray.wait(pending, timeout=0.5)
                
                if len(pending) > 0:
                    elapsed = int(time.time() - start_wait)
                    status = "⚡ Calculating"

                    if elapsed > guideline_limit:
                        advice = f"⚠️ Too Long! (> {guideline_limit}s). Consider Stop."
                    else:
                        advice = f"(Limit: ~{guideline_limit}s)"

                    spinner = ["|", "/", "-", "\\"][elapsed % 4]
                    sys.stdout.write(f"\r{spinner} {status} | Time: {elapsed}s {advice} | Waiting: {len(pending)} ")
                    sys.stdout.flush()
            
            # --- 完了 ---
            results = ray.get(futures)
            total_loss = sum(r['loss'] for r in results) / len(results)
            
            # 完了ログ (VRAM節約モードなので爆速のはず)
            print(f"\r✅ Step {step:04d} | Loss: {total_loss:.4f} | Time: {int(time.time()-start_wait)}s {' '*30}")

    except KeyboardInterrupt:
        print("\n\n🛑 STOP: User stopped training.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    start_training()