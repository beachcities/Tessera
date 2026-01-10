
import os
import torch
import requests
import numpy as np

# Tiny ShakespeareのURL
DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
DATA_PATH = os.path.join('data', 'input.txt')

class TextLoader:
    def __init__(self, block_size, batch_size, device='cuda'):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        
        # 1. データの準備 (なければダウンロード)
        if not os.path.exists(DATA_PATH):
            print(f"📥 Downloading {DATA_URL}...")
            try:
                with open(DATA_PATH, 'w') as f:
                    f.write(requests.get(DATA_URL).text)
            except Exception as e:
                print(f"❌ Download failed: {e}")
                raise
        
        # 2. 読み込み
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            self.text = f.read()
            
        print(f"📄 データ読み込み完了: {len(self.text)} 文字")

        # 3. トークナイザー構築 (文字レベル)
        chars = sorted(list(set(self.text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
        print(f"🔡 ボキャブラリサイズ: {self.vocab_size} (ユニークな文字数)")

        # 4. 全データをテンソル化
        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        
        # 訓練データ(90%) と 検証データ(10%) に分割
        n = int(0.9 * len(self.data))
        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        # デバイスへ転送 (エラー回避のため、device引数が有効か確認)
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available. Using CPU.")
            self.device = 'cpu'
            
        return x.to(self.device), y.to(self.device)

if __name__ == '__main__':
    # テスト実行
    print("🧪 Testing TextLoader...")
    try:
        # デバイスは自動判定
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        loader = TextLoader(block_size=8, batch_size=4, device=device)
        
        x, y = loader.get_batch('train')
        print("\n--- Batch Sample ---")
        print(f"Input shape: {x.shape}")
        print(f"Decoded Input: '{loader.decode(x[0].tolist())}'")
        print("✅ Success! Data pipeline is ready.")
    except Exception as e:
        print(f"❌ Error during test: {e}")
