
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from mamba_ssm import Mamba
from src.data import TextLoader
import time

# --- ハイパーパラメータ設定 (A100用に調整) ---
batch_size = 64
block_size = 256    # 文脈の長さ
max_iters = 1000    # 学習ステップ数 (デモ用なので少なめ)
eval_interval = 100 # ログ出力間隔
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# モデル設定 (Smallモデル相当)
d_model = 256
d_state = 16
d_conv = 4
expand = 2

def main():
    print(f"🚀 Mamba学習開始 (Device: {device})")
    
    # 1. データローダーの準備
    loader = TextLoader(block_size=block_size, batch_size=batch_size, device=device)
    vocab_size = loader.vocab_size
    print(f"📚 データ準備完了: Vocab size = {vocab_size}")

    # 2. モデル定義
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
            # 推論(生成)モード
            for _ in range(max_new_tokens):
                logits = self(idx)
                logits = logits[:, -1, :] # 最後の文字の予測だけ使う
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx

    model = MambaLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    print("🤖 モデル構築完了: 学習ループに入ります...")

    # 3. 学習ループ
    start_time = time.time()
    model.train()
    
    for iter in range(max_iters):
        # バッチ取得
        xb, yb = loader.get_batch('train')

        # 順伝播・逆伝播
        logits = model(xb)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), yb.view(B*T))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ログ出力
        if iter % eval_interval == 0:
            print(f"step {iter}: loss {loss.item():.4f}")

    end_time = time.time()
    print(f"✅ 学習完了！ (所要時間: {end_time - start_time:.2f}秒)")

    # 4. 生成デモ (Inference)
    print("\n🖋️ 生成テスト: Mambaが書くシェイクスピア...")
    print("-" * 50)
    
    context = torch.zeros((1, 1), dtype=torch.long, device=device) # 0からスタート
    generated_ids = model.generate(context, max_new_tokens=200)
    print(loader.decode(generated_ids[0].tolist()))
    
    print("-" * 50)

if __name__ == '__main__':
    main()
