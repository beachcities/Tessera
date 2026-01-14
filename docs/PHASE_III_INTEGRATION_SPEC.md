# Phase III 統合仕様書（暫定）

**Date:** 2026-01-15
**Status:** 設計中

---

## 1. 統合方式

**後期融合（案2）を採用**

- Mamba の最終層直前で concat
- 既存の学習済み知能を維持しつつ、物理場を補正として注入

---

## 2. チャンネル設計

| コンポーネント | チャンネル数 | 役割 |
|----------------|-------------|------|
| Mamba 出力 | 128 | 主：時系列・トポロジー記憶 |
| RayCast 出力 | 16 | 副：長距離幾何学的直感 |
| concat 後 | 144 | 統合特徴量 |
| 1×1 conv 後 | 128 | 元のチャンネル数に戻す |

---

## 3. 正規化

- concat 直後に **LayerNorm**
- スケール差を吸収し、学習初期の暴走を防止

---

## 4. 距離減衰の実験条件

| 条件 | 減衰則 | 期待される効果 |
|------|--------|---------------|
| A | 1/d | 遠方まで影響、模様・厚み向き |
| B | 1/d² | 急減衰、シチョウ・局所戦向き |

---

## 5. 統合コード骨格
```python
class TesseractFusion(nn.Module):
    """
    Mamba 出力 + Tesseract Field の後期融合
    """
    def __init__(self, mamba_ch=128, ray_ch=16):
        super().__init__()
        self.norm = nn.LayerNorm([mamba_ch + ray_ch, 19, 19])
        self.conv = nn.Conv2d(mamba_ch + ray_ch, mamba_ch, kernel_size=1)
    
    def forward(self, mamba_out, ray_out):
        # mamba_out: [B, 128, 19, 19]
        # ray_out:   [B, 16, 19, 19]
        x = torch.cat([mamba_out, ray_out], dim=1)  # [B, 144, 19, 19]
        x = self.norm(x)
        x = self.conv(x)  # [B, 128, 19, 19]
        return x
```

---

## 6. 次のステップ

1. [ ] 1M ゲーム学習完了確認
2. [ ] `src/fusion_layer.py` 実装
3. [ ] 統合版モデル作成
4. [ ] 1/d vs 1/d² 比較実験（各 1-2 万ゲーム）
5. [ ] 可視化（シチョウ / 大模様 / コウ）

---

*Phase II の知能を継承し、物理場で拡張する*
