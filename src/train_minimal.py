
import torch
import torch.nn as nn
import torch.optim as optim
from mamba_ssm import Mamba

def main():
    print("🚀 Sanity Check: Mamba学習ループの動作確認を開始します...")
    
    # --- 設定 (A100ならもっと大きくできますが、動作確認なので軽めに) ---
    batch_size = 4
    seq_len = 128
    d_model = 64
    d_state = 16
    vocab_size = 100
    device = "cuda"

    print(f"⚙️ 設定: Batch={batch_size}, SeqLen={seq_len}, ModelDim={d_model}")

    # --- 1. モデル定義 (Embedding + Mamba + Head) ---
    class MambaLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=4,
                expand=2
            )
            self.head = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            x = self.mamba(x)
            logits = self.head(x)
            return logits

    model = MambaLM().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("✅ モデル構築完了")

    # --- 2. ダミーデータ生成 ---
    # ランダムな整数列を入力とします
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)

    # --- 3. 学習ループ (5ステップだけ回す) ---
    print("\n📉 学習開始 (Lossが下がるか確認)...")
    model.train()
    
    for step in range(1, 6):
        optimizer.zero_grad()
        
        # 順伝播
        logits = model(inputs)
        
        # Loss計算 (Flattenして渡す)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        
        # 逆伝播
        loss.backward()
        optimizer.step()
        
        print(f"   Step {step}: Loss = {loss.item():.4f}")

    print("\n✅ 成功: エラー落ちせず、勾配計算(Backward)が機能しています。")

if __name__ == "__main__":
    main()
