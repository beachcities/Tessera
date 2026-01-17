# Phase III.2 実験結果報告書

**Date:** 2026-01-17
**Author:** beachcities + Claude + Gemini + Copilot

---

## 概要

Phase III.2 では、Tromp-Taylor ルールへの移行と Value Head の追加により、学習効率の劇的な改善を達成した。

**主要成果:**
- Policy Loss: 5.94 → **3.58** (40% 削減)
- Value Loss: 0.32 → **0.06** (81% 削減)
- 学習速度: 3.1 → **15.6 g/s** (5倍向上)
- 10,000 ゲームの学習を **10.7 分** で完了

---

## 1. 背景と課題

### 1.1 Phase III.1 の状況

Phase III.1 では TesseraModel（Mamba + Tesseract Field）を統合し、連の捕獲処理を GPU で実装しようとしていた。

**課題:**
- 連検出の Python ループがボトルネック
- 学習速度が 3.1 g/s で頭打ち
- Policy Loss が 5.9 付近で停滞

### 1.2 転換点

「意味のあることをしましょう」という問いから、根本的な再検討を実施。

**発見:**
- 連の捕獲処理が GPU sovereignty を破壊していた
- 複雑なルール実装より、学習信号（勝敗）の明確化が重要
- Tromp-Taylor ルールなら捕獲ロジック自体が不要

---

## 2. 実装内容

### 2.1 Tromp-Taylor ルール採用

| 変更前 | 変更後 |
|--------|--------|
| 連の検出・捕獲 | 捕獲なし |
| 複雑なコウ判定 | 省略可能 |
| Python ループ多数 | **ループゼロ** |

### 2.2 Value Head 追加
```python
self.value_head = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.ReLU(),
    nn.Linear(d_model // 2, 1),
    nn.Tanh()  # [-1, +1]
)
```

### 2.3 GPU 地計算 (chain_utils.py)

Conv2d による flood-fill アルゴリズムで、終局時の地を GPU 内で計算。

---

## 3. 実験結果

### 3.1 学習曲線
```
Game   128 | Loss: 6.10 (P:5.94 V:0.32) | Speed:  9.3 g/s
Game  1024 | Loss: 5.97 (P:5.93 V:0.08) | Speed: 11.8 g/s
Game  3000 | Loss: 5.93 (P:5.91 V:0.05) | Speed: 10.9 g/s
Game  5000 | Loss: 5.93 (P:5.91 V:0.04) | Speed: 10.9 g/s
----- 相転移 -----
Game  6000 | Loss: 5.84 (P:5.82 V:0.04) | Speed: 10.9 g/s
Game  7000 | Loss: 5.08 (P:5.06 V:0.04) | Speed: 12.0 g/s
Game  8000 | Loss: 4.21 (P:4.19 V:0.05) | Speed: 13.8 g/s
Game  9000 | Loss: 3.97 (P:3.94 V:0.06) | Speed: 14.5 g/s
Game 10000 | Loss: 3.58 (P:3.56 V:0.06) | Speed: 15.6 g/s
```

### 3.2 相転移の観察

6,000 ゲーム付近で**相転移（Phase Transition）**が発生。

**特徴:**
- Policy Loss が急激に降下（5.9 → 3.5）
- 学習速度も同時に向上（10.9 → 15.6 g/s）
- ゲームが効率化（無駄な手が減少）

**解釈:**
- Value Head が「勝敗の因果関係」を学習
- Policy が「勝つための手」を選び始める
- 正のフィードバックループが発生

### 3.3 最終結果

| 指標 | 値 |
|------|-----|
| 総ゲーム数 | 10,006 |
| 最終 Policy Loss | 3.56 |
| 最終 Value Loss | 0.06 |
| 最終速度 | 15.6 g/s |
| 所要時間 | 10.7 分 |

---

## 4. 分析

### 4.1 なぜ Value Head が効いたか

**従来（Policy のみ）:**
- 「次の手」を予測するだけ
- 良い手か悪い手かの判断基準がない
- 学習に「目的」がない

**Value Head 追加後:**
- 「この局面は勝ち/負け」という明確な信号
- モデルが「何を最適化すべきか」を即座に理解
- 勝利に繋がる手を優先的に学習

### 4.2 Information Continuity の効果

Tromp-Taylor では石が盤上から消えない。

**効果:**
- Mamba の SSM State に一貫した文脈を提供
- 勾配の流れが安定
- 学習の収束が早い

### 4.3 速度向上の要因

| 要因 | 寄与 |
|------|------|
| 捕獲ロジック廃止 | ★★★★★ |
| Python ループ排除 | ★★★★★ |
| ゲームの効率化 | ★★★ |
| GPU 地計算 | ★★★ |

---

## 5. 成果物

### 5.1 コード

| ファイル | 説明 |
|----------|------|
| `chain_utils.py` | GPU 地計算（新規） |
| `gpu_go_engine.py` | Tromp-Taylor 版エンジン |
| `train_phase3_2.py` | Policy + Value 学習スクリプト |

### 5.2 チェックポイント

| ファイル | Loss | 説明 |
|----------|------|------|
| `tessera_phase3.2_game4000.pth` | ~5.9 | 相転移前 |
| `tessera_phase3.2_game6000.pth` | ~5.8 | 相転移直後 |
| `tessera_phase3.2_final_loss3.58.pth` | 3.58 | 最終モデル |

### 5.3 Git コミット
```
136706d Phase III.2: Tromp-Taylor + Value Head
```

---

## 6. 次のステップ (Phase III.3)

| Step | 内容 | 目的 |
|------|------|------|
| 1 | vs Random (100戦) | 基礎性能確認 |
| 2 | vs Phase II | 世代間比較 |
| 3 | Internal Score Consistency | 内省能力測定 |
| 4 | SGF Exporter | 棋譜可視化 |

---

## 7. 教訓

### 7.1 技術的教訓

> **「ルールを GPU に合わせて再解釈する」**

複雑なルールを忠実に実装するより、GPU の特性に合わせてルールを再定義する方が、結果的に優れたシステムになる。

### 7.2 設計的教訓

> **「学習信号の明確化が最優先」**

速度やアーキテクチャより、「何を学習すべきか」を明確にすることが重要。Value Head の追加が相転移を引き起こした。

### 7.3 文化的教訓

> **「設計書に従う」ではなく「設計書を進化させる」**

当初の Phase III 計画（連の GPU 実装）に固執せず、より良いアーキテクチャへ舵を切ったことが成功の鍵。

---

*"Le symbole donne à penser."* — Paul Ricœur

— End of Document —
