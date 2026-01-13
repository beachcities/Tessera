# Tessera Handoff Document

## Phase II-b 完了時点 (2026-01-13)

---

## プロジェクト概要

**Tessera** は Mamba (State Space Model) を用いた囲碁AIプロジェクト。
GPU-Native 設計により、全ての演算をGPU上で完結させる。

---

## 現在の状態

### 達成済み

| 項目 | 状態 |
|------|------|
| 100,000ゲーム学習 | ✅ 完了 |
| OOM問題 | ✅ 解決 (MambaStateCapture無効化) |
| ELO評価システム | ✅ 稼働 |
| 並列化学習 | ✅ BATCH_SIZE=64 |
| Graceful Shutdown | ✅ 実装済み |
| Discord監視 | ✅ 設定済み |

### 最終メトリクス

```
Best Loss: 1.5299
Final ELO: 1512
Speed: 38,000 games/hr
Total Time: 2.63 hours
```

---

## ファイル構成

```
GoMamba_Local/
├── src/
│   ├── gpu_go_engine.py    # GPU Native 囲碁エンジン (v0.2.2)
│   ├── model.py            # Mamba モデル (v0.2.2)
│   ├── elo.py              # ELO評価システム (v1.3.0)
│   ├── long_training_v4.py # 並列化学習 (v4.1.0)
│   └── monitor.py          # VRAMモニタリング
├── checkpoints/
│   └── tessera_v4_final_game100000_elo1512.pth  # 最終モデル
├── logs/
│   ├── training_v4_*.log   # 学習ログ
│   └── elo_*.jsonl         # ELO記録
├── docs/
│   ├── DESIGN_SPEC_PHASE_II.md
│   └── EXPERIMENT_LOG_20260113.md
├── monitor.sh              # Discord監視スクリプト
├── CHANGELOG.md
└── docker-compose.yml
```

---

## 重要な設計原則

### 1. GPU Sovereignty

```
VRAM は計算専用の聖域。
計算に直結しないデータの持ち込みを禁ずる。
```

### 2. CPU Staging

```python
# 正しい: CPU経由でロード
checkpoint = torch.load(filepath, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

# 間違い: 直接GPUにロード（Optimizer stateがVRAMを汚染）
checkpoint = torch.load(filepath, map_location='cuda')
```

### 3. Clean Room Protocol

```python
with VRAMSanitizer("ELO_Match"):
    # ここはクリーンな状態
    result = evaluator.play_match(...)
# 自動的にクリーンアップ
```

---

## 既知の問題・制約

### Phase II の制約

| 項目 | 状態 | Phase III で対応 |
|------|------|-----------------|
| 連の捕獲 | 単石のみ | Connected Components |
| 地の計算 | なし | 実装予定 |
| ELO対戦相手 | 自己対戦のみ | GnuGo/KataGo |
| コウ判定 | 簡易版 | スーパーコウ |

### 注意点

1. **MambaStateCapture は無効化されている**
   - 有効化する場合は必ず `clear()` を呼ぶ
   - または Context Manager で囲む

2. **ELO評価はメモリ負荷が高い**
   - Clean Room Protocol を必ず遵守
   - 学習用Engineを解放してから評価

---

## 起動方法

### 通常起動

```bash
cd ~/GoMamba_Local
docker compose up -d
docker compose exec tessera bash -c "cd /app && python3.10 src/long_training_v4.py"
```

### 再開（チェックポイントから）

```bash
docker compose exec tessera bash -c "cd /app && python3.10 src/long_training_v4.py --resume checkpoints/tessera_v4_final_game100000_elo1512.pth"
```

### 監視スクリプト

```bash
nohup ~/GoMamba_Local/monitor_simple.sh > ~/GoMamba_Local/monitor.log 2>&1 &
```

---

## 次のステップ

### Phase II-c (オプション)

- [ ] BATCH_SIZE=128 の検証
- [ ] Streamlit による可視化
- [ ] 学習曲線の分析

### Phase III

- [ ] Connected Components (連の実装)
- [ ] 地の計算
- [ ] GnuGo/KataGo との対戦
- [ ] 真のELO測定
- [ ] A100 60GB 環境での大規模学習

---

## 連絡先・参考

- Discord Webhook: 設定済み（monitor.sh内）
- 設計思想: `docs/DESIGN_SPEC_PHASE_II.md`
- 実験記録: `docs/EXPERIMENT_LOG_20260113.md`

---

*Last Updated: 2026-01-13*
*Phase II-b Complete*
