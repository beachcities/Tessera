
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from mamba_ssm import Mamba
from src.data import TextLoader
import time
import os
import datetime

# --- 設定 ---
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = "checkpoints"
MODEL_NAME = f"mamba_shakespeare_{TIMESTAMP}.pth"

batch_size = 64
block_size = 256
max_iters = 500   # テスト用に500回に短縮
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# モデルパラメータ
d_model = 256
d_state = 16
d_conv = 4
expand = 2

def save_checkpoint(model, optimizer, iter_num, loss, filepath):
    print(f"💾 Saving checkpoint to {filepath}...")
    torch.save({
        'iter': iter_num,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)

def main():
    print(f"🚀 Mamba学習開始 (Device: {device})")
    print(f"📂 Save target: {os.path.join(CHECKPOINT_DIR, MODEL_NAME)}")
    
    loader = TextLoader(block_size=block_size, batch_size=batch_size, device=device)
    vocab_size = loader.vocab_size

    class MambaLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.mamba = Mamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, idx):
            x = self.embedding(idx)
            x = self.mamba(x)
            logits = self.lm_head(x)
            return logits

        def generate(self, idx, max_new_tokens):
            for _ in range(max_new_tokens):
                logits = self(idx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    model = MambaLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    start_time = time.time()
    model.train()
    
    for iter in range(max_iters):
        xb, yb = loader.get_batch('train')
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % eval_interval == 0:
            print(f"step {iter}: loss {loss.item():.4f}")

    end_time = time.time()
    print(f"✅ 学習完了！ (所要時間: {end_time - start_time:.2f}秒)")

    # 保存
    save_path = os.path.join(CHECKPOINT_DIR, MODEL_NAME)
    save_checkpoint(model, optimizer, max_iters, loss.item(), save_path)
    
    # 生成デモ
    print("\n🖋️ 生成テスト:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=200)
    print(loader.decode(generated_ids[0].tolist()))

if __name__ == '__main__':
    main()
