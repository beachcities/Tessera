# Tessera (MambaGo) 引き継ぎドキュメント

**Date:** 2026-01-12
**Status:** Phase II 完了

---

## 現在の状態

### 完了済み

| Phase | 内容 | 状態 |
|-------|------|------|
| I | 環境構築（Docker + CUDA 12.6 + Mamba SSM） | ✅ |
| II | GPU-Native Go Engine + MambaModel統合 | ✅ |

### 動作確認済みコンポーネント

| ファイル | 役割 | テスト |
|----------|------|--------|
| `src/monitor.py` | TesseraMonitor（VRAM, SSM State監視） | ✅ |
| `src/gpu_go_engine.py` | GPUGoEngine（バッチ着手、単石捕獲） | ✅ |
| `src/model.py` | MambaModel（4層、2Mパラメータ） | ✅ |
| `src/full_integration_test.py` | 自己対局+学習ループ | ✅ |

---

## 環境起動手順
```bash
cd ~/GoMamba_Local
docker compose up -d
docker compose exec tessera bash

# コンテナ内で
python3.10 src/full_integration_test.py
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
│  MambaModel (4-layer SSM)               │
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

## Phase II 制限事項（Phase III で対応）

1. **単石捕獲のみ** - 連の実装なし
2. **単純コウのみ** - スーパーコウなし
3. **自殺手判定が不完全** - 単石のみ

---

## 設計仕様書

`docs/DESIGN_SPEC_PHASE_II.md` を参照

---

## 次のステップ（Phase III 候補）

1. **長期学習テスト** - 100ゲーム以上
2. **連のGPU実装** - Connected Components
3. **Incremental Inference** - 1手ずつ状態更新
4. **Loss減少の確認** - 現在5.89、目標は継続的減少

---

## 重要なファイルパス

- 設計仕様: `docs/DESIGN_SPEC_PHASE_II.md`
- Docker: `docker-compose.yml`, `Dockerfile`
- ソース: `src/monitor.py`, `src/gpu_go_engine.py`, `src/model.py`
- 旧コード: `main360_gpu_v4.py`（参考用）

---

## 思想（The Mythos）

> MambaGoは命令しない。確率分布という「可能性の地図」を示す。
> 最後の一手は常にユーザーが選ぶ（Agency）。

*"Le symbole donne à penser."* — Paul Ricœur

---

## コマンドチートシート
```bash
# 環境起動
docker compose up -d
docker compose exec tessera bash

# テスト実行
python3.10 src/monitor.py
python3.10 src/gpu_go_engine.py
python3.10 src/model.py
python3.10 src/full_integration_test.py

# Git操作
git add -A
git commit -m "message"
git push origin main
```

---

*The Serpent awaits.*
