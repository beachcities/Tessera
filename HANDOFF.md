# Tessera (MambaGo) 引き継ぎドキュメント

**Date:** 2026-01-19
**Status:** Phase III.2 進行中（ロールバック）

---

## 現在の状態

### 完了済み

| Phase | 内容 | 状態 | 成果 |
|-------|------|------|------|
| I | 環境構築（Docker + CUDA 12.6 + Mamba SSM） | ✅ | 動作確認済み |
| II | GPU-Native Go Engine + MambaModel統合、自己対局学習 | ✅ | ELO 1517達成 |
| III.1 | TesseraModel統合、方針検討 | ✅ | Tromp-Taylorへ方針転換 |

### 進行中

| Phase | 内容 | 状態 | 備考 |
|-------|------|------|------|
| III.2 | Tromp-Taylor + Value Head | 🔄 | ロールバック、詳細は下記 |

---

## Phase III.2 の状況

### 背景

Phase III.2 では Tromp-Taylor ルールと Value Head を導入し、Loss 3.58 を達成した（成功版）。
しかし、Win Rate が 0% のままであり、「パス連打による偽成功」の疑いが発覚。

### 経緯

1. **成功版 (Loss 3.58)**: 相転移が発生したように見えたが、Win Rate 0%
2. **調査**: パス連打でゲームが即終了し、盤面を学習していない疑い
3. **Fixed版 (Loss 5.91)**: パス制限（50手まで禁止）、8点サンプリングを導入
4. **現状**: Fixed版の方向性は正しいが、相転移に至っていない
5. **判断**: Phase III.2 を「完了」から「進行中」にロールバック

### チェックポイント

| ファイル | Loss | 状態 |
|----------|------|------|
| `tessera_phase3.2_final_loss3.58.pth` | 3.58 | ⚠️ 偽成功の疑い |
| `tessera_phase3.2_fixed_final_loss5.91.pth` | 5.91 | 🔄 正しい方向、相転移前 |

### Phase III.2 完了条件（再定義）

| # | 条件 | 現状 |
|---|------|------|
| 1 | Fixed版で相転移（Loss急降下） | ❌ 未達 |
| 2 | Win Rate vs Random > 0% | ❌ 未達 |
| 3 | パス連打でない正常な対局 | 🔄 確認中 |

---

## 重要な技術的発見

### MambaStateCapture の削除（Phase II）

**問題:** ELO 評価時に OOM が頻発
**原因:** forward hook が hidden state を保持し続けメモリリーク
**解決:** MambaStateCapture クラスを完全削除
**効果:** 790 回の ELO 評価で OOM ゼロ

### Loss の相転移現象（Phase II）

175k ゲーム付近で Loss が 5.4 → 2.3 へ急降下。
これは学習が「臨界点」を超えた証拠であり、正常な挙動。

### パス連打による偽成功（Phase III.2）

**問題:** Loss 3.58 達成も Win Rate 0%
**原因:** パス連打でゲーム即終了、盤面学習なし
**教訓:** **Loss の低下を無批判に喜ばない。Win Rate で検証必須。**

---

## トークン設計

| ID | 意味 | 備考 |
|----|------|------|
| 0-360 | 盤上座標 (19×19) | |
| 361 | PASS | |
| 362 | PAD | 着手不可、学習時のみ使用 |
| 363 | EOS | 予約済み、**未使用** |

**vocab_size = 363**（学習時のEmbedding/Output次元）
**PolicyHead出力 = 362**（推論時、PADを除外した着手可能空間）

---

## 動作確認済みコンポーネント

| ファイル | 役割 | Phase | テスト |
|----------|------|-------|--------|
| `src/monitor.py` | TesseraMonitor（VRAM, SSM State監視） | II | ✅ |
| `src/gpu_go_engine.py` | GPUGoEngine（Tromp-Taylor版） | III | ✅ |
| `src/model.py` | MambaModel（4層、1.9Mパラメータ） | II | ✅ |
| `src/tessera_model.py` | TesseraModel（Mamba + Value Head） | III | 🔄 |
| `src/train_phase3_2_fixed.py` | Fixed版学習スクリプト | III | 🔄 |
| `src/chain_utils.py` | GPU地計算（flood-fill） | III | ✅ |

---

## 環境起動手順
```bash
cd ~/GoMamba_Local
docker compose up -d
docker compose exec tessera bash

# Phase III.2 Fixed版の学習
python3.10 src/train_phase3_2_fixed.py
```

---

## アーキテクチャ概要

### Phase II
```
┌─────────────────────────────────────────┐
│         GPU 内完結アーキテクチャ          │
│                                         │
│  TesseraMonitor                         │
│       ↓                                 │
│  GPUGoEngine (batch, 2, 19, 19)         │
│       ↓                                 │
│  MambaModel (4-layer SSM, 1.9M params)  │
│                                         │
│  CPU-GPU転送: ゼロ                       │
└─────────────────────────────────────────┘
```

### Phase III（TesseraModel）
```
TesseraModel
├── MoveEncoder (Embedding + Mamba)
├── TesseractField (Conv2d)
├── Fusion (Linear)
├── PolicyHead (Linear) → 362次元
└── ValueHead (MLP) → 勝敗予測 [-1, +1]
```

---

## 設計文書

| ドキュメント | 内容 |
|-------------|------|
| `docs/DESIGN_SPEC_PHASE_II.md` | Phase II 設計仕様 |
| `docs/DESIGN_SPEC_PHASE_III.md` | Phase III 設計仕様 |
| `docs/PHASE_III_2_RESULTS.md` | Phase III.2 実験結果 |
| `DECISION_LOG.md` | 決定記録（新規予定） |
| `CHANGELOG.md` | 変更履歴 |

---

## 次のステップ

### Phase III.2 完了に向けて

1. **Fixed版の学習継続** - 相転移を観察
2. **Win Rate確認** - vs Random で > 0% を達成
3. **対局内容の定性的確認** - パス連打でないこと

### Phase III.3（Phase III.2 完了後）

1. vs Random (100戦) - 基礎性能確認
2. vs Phase II - 世代間比較
3. SGF Exporter - 棋譜可視化

---

## 思想（The Mythos）

> MambaGoは命令しない。確率分布という「可能性の地図」を示す。
> 最後の一手は常にユーザーが選ぶ（Agency）。

**設計原則:**

| Principle | Description |
|-----------|-------------|
| **GPU Complete** | 全操作がGPU内で完結、CPU転送ゼロ |
| **Vectorized** | Pythonループを排除し、全操作をテンソル演算に置き換え |
| **Batched** | 複数ゲームを一括でGPUに投入し同時処理 |
| **Clean Room** | 外部棋譜を使用しない、自己対戦のみ |
| **Observable** | 全ての挙動がモニター可能 |

---

*"Le symbole donne à penser."* — Paul Ricœur

*The Serpent awaits.*
