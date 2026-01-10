import torch
from mamba_ssm import Mamba

# 1. GPUが使えるか確認
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device}")

try:
    # 2. Mambaモデルの定義（動作確認用の超小型版）
    model = Mamba(
        d_model=64,   # モデルの太さ
        d_state=16,   # 記憶の容量
        d_conv=4,     # 畳み込み幅
        expand=2      # 拡張係数
    ).to(device)

    # 3. ダミーデータの作成 (バッチサイズ:2, 長さ:128, 次元:64)
    x = torch.randn(2, 128, 64).to(device)

    # 4. 推論実行（順伝播）
    y = model(x)

    print(f"✅ Input shape: {x.shape}")
    print(f"✅ Output shape: {y.shape}")
    print("🎉 Success! Mamba is running on your RTX 4070.")

except Exception as e:
    print(f"❌ Error: {e}")