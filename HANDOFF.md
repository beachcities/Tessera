# Tessera (MambaGo) 引き継ぎドキュメント

**Date:** 2026-01-22
**Status:** Phase III.2 完了、Phase III.3 準備中

---

## 現在の状態

### 完了済み

| Phase | 内容 | 状態 | 成果 |
|-------|------|------|------|
| I | 環境構築（Docker + CUDA 12.6 + Mamba SSM） | ✅ | 動作確認済み |
| II | GPU-Native Go Engine + MambaModel統合、自己対局学習 | ✅ | ELO 1517達成 |
| III.1 | TesseraModel統合、方針検討 | ✅ | Tromp-Taylorへ方針転換 |
| III.2 | Tromp-Taylor + Value Head + 高速化 | ✅ | **相転移達成、Win Rate > 0%** |

### 準備中

| Phase | 内容 | 状態 | 備考 |
|-------|------|------|------|
| III.3 | Value-Guided Policy Improvement | ⏳ | DEC-010で方針策定済み |

---

## Phase III.2 の成果（2026-01-22 完了）

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

### チェックポイント

| ファイル | Loss | 状態 |
|----------|------|------|
| `tessera_phase3.2_fixed_final_loss4.41.pth` | 4.41 | ✅ **Phase III.2 完了版** |
| `tessera_phase3.2_fixed_final_loss5.91.pth` | 5.91 | 旧版（相転移前） |
| `tessera_phase3.2_final_loss3.58.pth` | 3.58 | ⚠️ 偽成功（パス連打） |

### 貢献した決定

- **DEC-008**: 視点正規化（current_board * perspective）
- **DEC-009**: Turn Embedding（自分/相手の識別）
- **DEC-011**: VectorizedGameHistory（履歴管理のテンソル化）
- **#16**: replay_history_to_boards_fast GPU化（One-Hot + Cumsum方式）

---

## 重要な技術的発見

### 相転移の観測（Phase III.2）

- Policy Loss < 5.1（Game 31744付近）から Win Rate > 0% が安定出現
- 初期の散発的勝利（Game 1024, 14080）はノイズ
- Game 31744以降の勝利は「学習成果」

### GPU化による高速化（Phase III.2）

- VectorizedGameHistory: Pythonループ排除、アトミック管理
- One-Hot + Cumsum: replay_history_to_boards_fastの完全ベクトル化
- 結果: 1.9 g/s → 5.1 g/s（2.7倍）

### パス連打による偽成功（Phase III.2 初期）

**問題:** Loss 3.58 達成も Win Rate 0%
**原因:** パス連打でゲーム即終了、盤面学習なし
**教訓:** **Loss の低下を無批判に喜ばない。Win Rate で検証必須。**

### MambaStateCapture の削除（Phase II）

**問題:** ELO 評価時に OOM が頻発
**原因:** forward hook が hidden state を保持し続けメモリリーク
**解決:** MambaStateCapture クラスを完全削除

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
| `src/train_phase3_2_fixed.py` | Fixed版学習スクリプト v0.3.0 | III | ✅ |
| `src/chain_utils.py` | GPU地計算（flood-fill） | III | ✅ |
| `src/utils.py` | get_turn_sequence等のユーティリティ | III | ✅ |
| `src/eval_quick.py` | 簡易評価（vs Random） | III | ✅ |

---

## 環境起動手順
```bash
cd ~/GoMamba_Local
docker compose up -d
docker compose exec tessera bash

# Phase III.2 完了版の評価
python3.10 -c "
import torch
from tessera_model import TesseraModel
from eval_quick import quick_eval

model = TesseraModel().to('cuda')
model.load_state_dict(torch.load('/app/checkpoints/tessera_phase3.2_fixed_final_loss4.41.pth'), strict=False)
win_rate = quick_eval(model, device='cuda', num_games=256, verbose=True)
"
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

### 学習ループ（v0.3.0）
```
VectorizedGameHistory (Preallocated Tensor)
       ↓
GPUGoEngine.play_batch()
       ↓
replay_history_to_boards_fast() [One-Hot + Cumsum]
       ↓
TesseraModel.forward() [Policy + Value]
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
| `docs/KNOWN_TRAPS.md` | 既知の罠（TRAP-001〜009） |
| `docs/PARKING_LOT.md` | 保留事項と完了事項 |
| `docs/IMPLEMENTATION_PRINCIPLES.md` | 実装原則（Copilot策定） |
| `DECISION_LOG.md` | 決定記録（DEC-001〜012） |

---

## 次のステップ

### Phase III.3（DEC-010 参照）

1. **Advantage導入** - Policy Lossに勝敗重み付け
2. **温度調整** - 1.5 → 2.5 で探索多様性向上
3. **サンプル数増加** - 8 → 16

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
