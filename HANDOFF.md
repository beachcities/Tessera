# Tessera (MambaGo) 引き継ぎドキュメント

**Date:** 2026-01-22
**Status:** Phase III.3 本番学習進行中（v0.3.2）

---

## 現在の状態

### 完了済み

| Phase | 内容 | 状態 | 成果 |
|-------|------|------|------|
| I | 環境構築（Docker + CUDA 12.6 + Mamba SSM） | ✅ | 動作確認済み |
| II | GPU-Native Go Engine + MambaModel統合、自己対局学習 | ✅ | ELO 1517達成 |
| III.1 | TesseraModel統合、方針検討 | ✅ | Tromp-Taylorへ方針転換 |
| III.2 | Tromp-Taylor + Value Head + 高速化 | ✅ | **相転移達成、Win Rate > 0%** |

### 進行中

| Phase | 内容 | 状態 | 備考 |
|-------|------|------|------|
| III.3 | Value-Guided Policy Improvement | 🔄 | v0.3.2で本番学習中 |

---

## Phase III.3 の状態（2026-01-22）

### 本番学習進行中

| 項目 | 値 |
|------|-----|
| バージョン | v0.3.2（Corrected & Stabilized） |
| 目標ゲーム数 | 100,000 |
| 速度 | 12.4 g/s（Phase III.2の2.4倍） |
| 開始チェックポイント | tessera_phase3.2_fixed_final_loss4.41.pth |

### v0.3.2 で修正されたバグ

| バグ | 症状 | 原因 | 修正 |
|------|------|------|------|
| Actor-Critic発散 | PG Loss負の無限大 | 負のAdvantageで勾配爆発 | Positive Advantage Masking |
| Off-by-One Error | Win Rate 0%継続 | board_indices = sample_indices - 1 | board_indices = sample_indices |
| 形状エラー | IndexError | 不要なunsqueeze(1) | 削除 |
| Logits次元エラー | IndexError | policy_logits[:, -1, :] | policy_logits |

### 初期ログ（Game 1024時点）

| 指標 | 値 | 評価 |
|------|-----|------|
| PG Loss | 1.49 | ✅ 正の値（発散していない） |
| CE Loss | 5.44 | ✅ 正常範囲 |
| Entropy | 5.15 | ✅ 探索段階として健全 |
| Win Rate | 0.0% | ⏳ 相転移前（想定内） |

### Phase III.3 完了条件

| # | 条件 | 現状 |
|---|------|------|
| 1 | 発散なし（CE < 10） | ✅ 達成（CE 5.44） |
| 2 | Win Rate vs Random > 0% | ⏳ 未達（学習中） |
| 3 | Phase III.2 以上の性能 | ⏳ 未確認 |

---

## Phase III.2 の成果（参考）

### 達成した完了条件

| # | 条件 | 結果 |
|---|------|------|
| 1 | 相転移（Loss急降下） | ✅ Policy Loss 5.89→4.36（-1.53） |
| 2 | Win Rate vs Random > 0% | ✅ 後半で複数回達成（3.1%×3回、1.6%×1回） |
| 3 | パス連打でない正常な対局 | ✅ Pass率 0.0-0.2% |

### 技術的成果

| 項目 | Before | After | 改善 |
|------|--------|-------|------|
| 学習速度 | 1.9 g/s | 4.7-5.1 g/s | +168% |
| Policy Loss | 5.89 | 4.36 | -1.53 |
| Total Loss | 6.30 | 4.41 | -1.89 |

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
| `src/gpu_go_engine.py` | GPUGoEngine（Tromp-Taylor版、GPU化済み） | III | ✅ |
| `src/model.py` | MambaModel（4層、1.9Mパラメータ） | II | ✅ |
| `src/tessera_model.py` | TesseraModel（Mamba + Value Head） | III | ✅ |
| `src/train_phase3_2_fixed.py` | Phase III.2 学習スクリプト v0.3.0 | III | ✅ |
| `src/train_phase3_3.py` | Phase III.3 学習スクリプト v0.3.2 | III | ✅ |
| `src/debug_batch_semantics.py` | Semantic Sanity Check | III | ✅ |
| `src/chain_utils.py` | GPU地計算（flood-fill） | III | ✅ |
| `src/utils.py` | get_turn_sequence等のユーティリティ | III | ✅ |
| `src/eval_quick.py` | 簡易評価（vs Random） | III | ✅ |

---

## チェックポイント

| ファイル | Loss | 状態 |
|----------|------|------|
| `tessera_phase3.2_fixed_final_loss4.41.pth` | 4.41 | ✅ Phase III.2 完了版、III.3 開始点 |
| `tessera_phase3.2_fixed_final_loss5.91.pth` | 5.91 | 旧版（相転移前） |
| `tessera_phase3.2_final_loss3.58.pth` | 3.58 | ⚠️ 偽成功（パス連打） |

---

## 環境起動手順
```bash
cd ~/GoMamba_Local
docker compose up -d
docker compose exec tessera bash

# 学習状況確認
tail -f ~/GoMamba_Local/training_phase3_3.log

# Phase III.3 学習停止
docker compose exec tessera pkill -f train_phase3_3.py
```

---

## アーキテクチャ概要

### Phase III（TesseraModel）
```
TesseraModel
├── MoveEncoder (Embedding + Mamba + Turn Embedding)
├── TesseractField (Conv2d)
├── Fusion (Linear)
├── PolicyHead (Linear) → 362次元
└── ValueHead (MLP) → 勝敗予測 [-1, +1]
```

### Phase III.3 学習ループ（v0.3.2）
```
VectorizedGameHistory (Preallocated Tensor)
       ↓
GPUGoEngine.play_batch()
       ↓
replay_history_to_boards_fast() [One-Hot + Cumsum]
       ↓
TesseraModel.forward() [Policy + Value]
       ↓
Advantage計算 (Winner - Value)
       ↓
Positive Advantage Masking（負を除外）
       ↓
Loss計算 + Backward
```

---

## 設計文書

| ドキュメント | 内容 |
|-------------|------|
| `docs/DESIGN_SPEC_PHASE_II.md` | Phase II 設計仕様 |
| `docs/DESIGN_SPEC_PHASE_III.md` | Phase III 設計仕様 |
| `docs/PHASE_III_2_RESULTS.md` | Phase III.2 実験結果 |
| `docs/KNOWN_TRAPS.md` | 既知の罠（TRAP-001〜011） |
| `docs/PARKING_LOT.md` | 保留事項と完了事項 |
| `docs/IMPLEMENTATION_PRINCIPLES.md` | 実装原則（Active Shape Guarding追加） |
| `DECISION_LOG.md` | 決定記録（DEC-001〜013） |

---

## 重要な技術的発見

### Off-by-One Error（Phase III.3）

**問題:** バッチ化時に `board_indices = sample_indices - 1` としていた
**症状:** Win Rate 0% が継続（相手の最後の手が見えない状態で予測）
**検証:** `debug_batch_semantics.py` で盤面可視化
**修正:** `board_indices = sample_indices`
**教訓:** 形状が合っていても意味がズレていれば学習は進まない（Semantic Sanity Check）

### Actor-Critic Divergence（Phase III.3）

**問題:** 負のAdvantageで勾配爆発
**症状:** PG Loss負の無限大、CE > 23
**原因:** `∂/∂p(-ln p) = -1/p` は `p→0` で無限大
**修正:** Positive Advantage Masking（負のAdvantageを学習対象から除外）

---

## 次のステップ

### 学習完了後の確認事項

1. Win Rate > 0% の達成確認
2. Phase III.2 (Loss 4.41) との性能比較
3. チェックポイント保存

### 長期目標

1. vs Random (100戦) - 安定した勝率確認
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

---

## Phase III.3 更新（2026-01-24）

### 現在の状態

| 項目 | 状態 |
|------|------|
| Phase | III.3 Phase 1 完了 |
| バージョン | v3.4 (train_phase3_4_robust.py) |
| 達成 | 10,000ゲーム完走 |
| 最終モデル | tessera_phase3.3_final.pth |
| 次の目標 | Phase 2（20,000ゲーム） |

### 主要機能（v3.4）

- **Atomic Checkpoint Save**: 一時ファイル経由の安全な保存
- **Signal Handling**: SIGTERM安全停止
- **Surgical Gradient Scaling**: 特定層（x_proj）の個別クリップ
- **Data Guard**: Advantage Clipping（±10.0）、Per-sample Loss Cap（10.0）
- **Debug Context**: 緊急停止時の完全状態保存

### 最適パラメータ（v3.4）

| パラメータ | 値 |
|------------|-----|
| LEARNING_RATE | 2.5e-6 |
| BATCH_SIZE | 16 |
| GRADIENT_CLIP_NORM | 0.5 |
| PG_LOSS_CLIP | 4.0 |
| GUARD_STOP | 200.0 |
| GUARD_EMERGENCY | 150.0 |
| GUARD_WARN | 50.0 |

### チェックポイント

| ファイル | 用途 |
|----------|------|
| tessera_phase3.3_final.pth | Phase 1完了、Phase 2初期重み |
| archive_phase3.3/*.pth | クラッシュ時チェックポイント（解析用） |
