# Tessera (MambaGo) 引き継ぎドキュメント

**Date:** 2026-01-14
**Status:** Phase II-c 完了、Phase III 準備中

---

## 現在の状態

### 完了済み

| Phase | 内容 | 状態 | 成果 |
|-------|------|------|------|
| I | 環境構築（Docker + CUDA 12.6 + Mamba SSM） | ✅ | 動作確認済み |
| II-a | GPU-Native Go Engine + MambaModel統合 | ✅ | 自己対局成功 |
| II-b | 100,000 ゲーム学習 | ✅ | ELO 1512 |
| II-c | 200,000 ゲーム学習 | ✅ | ELO 1496、最高 1517 |

### 最新チェックポイント

```
checkpoints/tessera_v4_final_game200096_elo1496.pth
```

---

## 重要な技術的発見

### MambaStateCapture の削除

**問題:** ELO 評価時に OOM が頻発
**原因:** forward hook が hidden state を保持し続けメモリリーク
**解決:** MambaStateCapture クラスを完全削除
**効果:** 790 回の ELO 評価で OOM ゼロ

### Loss の相転移現象

175k ゲーム付近で Loss が 5.4 → 2.3 へ急降下。
これは学習が「臨界点」を超えた証拠であり、正常な挙動。

---

## 動作確認済みコンポーネント

| ファイル | 役割 | テスト |
|----------|------|--------|
| `src/monitor.py` | TesseraMonitor（VRAM, SSM State監視） | ✅ |
| `src/gpu_go_engine.py` | GPUGoEngine（バッチ着手、単石捕獲） | ✅ |
| `src/model.py` | MambaModel（4層、1.9Mパラメータ） | ✅ |
| `src/long_training_v4.py` | 長期学習（ELO評価、Tile保存） | ✅ |
| `streamlit_dashboard.py` | リアルタイム可視化 | ✅ |

---

## 環境起動手順

```bash
cd ~/GoMamba_Local
docker compose up -d
docker compose exec tessera bash

# コンテナ内で学習再開
python3.10 src/long_training_v4.py --resume checkpoints/tessera_v4_final_game200096_elo1496.pth
```

---

## アーキテクチャ概要

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

---

## トークン設計

| ID | 意味 |
|----|------|
| 0-360 | 盤上座標 (19×19) |
| 361 | PASS |
| 362 | PAD |
| 363 | EOS |
| **364** | VOCAB_SIZE |

---

## Phase II 制限事項（Phase III で対応予定）

1. **単石捕獲のみ** - 連の実装なし
2. **単純コウのみ** - スーパーコウなし
3. **自殺手判定が不完全** - 単石のみ
4. **終局判定が単純** - 二連続パスのみ、地の計算なし

---

## 設計文書

| ドキュメント | 内容 |
|-------------|------|
| `docs/DESIGN_SPEC_PHASE_II.md` | Phase II 設計仕様 |
| `docs/DESIGN_SPEC_PHASE_III.md` | Phase III 設計（新規） |
| `EXPERIMENT_LOG.md` | 実験結果の記録 |
| `CHANGELOG.md` | 変更履歴 |

---

## 次のステップ（Phase III 候補）

### 優先度 高
1. **連の GPU 実装** - Connected Components
2. **終局判定の改善** - 地の計算

### 優先度 中
3. **best_loss の修正** - 移動平均に変更
4. **外部評価** - GnuGo/KataGo との対戦（評価専用）

### 将来検討（未検証）
5. **19^4 テンソル設計** - 理論的検証が必要
6. **Incremental Inference** - 1手ずつ状態更新
7. **確定領域マスク** - 終盤の計算効率化

---

## Streamlit ダッシュボード起動

```bash
cd ~/GoMamba_Local
source venv/bin/activate
streamlit run streamlit_dashboard.py
# ブラウザで http://localhost:8501
```

---

## コマンドチートシート

```bash
# 環境起動
docker compose up -d
docker compose exec tessera bash

# 学習実行
python3.10 src/long_training_v4.py

# 学習再開
python3.10 src/long_training_v4.py --resume checkpoints/tessera_v4_final_game200096_elo1496.pth

# ログ確認
tail -f logs/training_v4_*.log

# Git操作
git add -A
git commit -m "message"
git push origin main
```

---

## 思想（The Mythos）

> MambaGoは命令しない。確率分布という「可能性の地図」を示す。
> 最後の一手は常にユーザーが選ぶ（Agency）。

**設計原則:**
- GPU Complete: 全操作が GPU 内で完結
- Batch First: 全操作がバッチ化
- Clean Room: 外部棋譜を使用しない、自己対戦のみ

---

*"Le symbole donne à penser."* — Paul Ricœur

*The Serpent awaits.*
