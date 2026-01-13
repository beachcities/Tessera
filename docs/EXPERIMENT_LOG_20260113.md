# Experiment Log: 2026-01-13

## Phase II-b: 長期学習テスト + 並列化 + ELO導入

---

## 概要

100,000ゲームの長期学習を完走。OOM問題を解決し、GPU-Native設計の有効性を実証。

---

## 最終結果

| 項目 | 値 |
|------|-----|
| **Games** | 100,000 ✅ |
| **Total Moves** | 18,645,417 |
| **Total Time** | 2.63 hours |
| **Best Loss** | 1.5299 |
| **Final ELO** | 1512 |
| **Best ELO** | 1516 |
| **Speed** | 38,000 games/hr |

---

## バージョン進化

| Version | 変更点 | 結果 |
|---------|--------|------|
| v1 | 初期実装 | Game 412 で SSM 発散 (Norm 529.7) |
| v3 | 安全装置追加 | 安定稼働、BATCH=16 で 2,600/hr |
| v3 (BATCH=64) | バッチサイズ増加 | 3回目ELOでOOM |
| v4.0 | 並列化 + ELO統合 | OOM継続（MambaStateCapture問題） |
| v4.1 | Clean Room Protocol | OOM継続 |
| **v4.1 + fix** | **MambaStateCapture無効化** | **100,000ゲーム完走** ✅ |

---

## 根本原因と解決

### OOMの真犯人: MambaStateCapture

```python
# 問題のコード (line 366)
self.state_capture = MambaStateCapture(self.model)
```

**症状:**
- 1回目ELO: 成功
- 2回目ELO: 成功
- 3回目ELO: OOM (21GB超過)

**原因:**
- MambaStateCapture が forward hook を登録
- 各推論の hidden state を Python list に蓄積
- clear() が一度も呼ばれず無限に成長
- VRAM → Unified Memory → システム限界 → OOM

**解決:**
```python
# 1行コメントアウト
#self.state_capture = MambaStateCapture(self.model)  # DISABLED: Memory leak
```

---

## 設計原則の確立

### 1. GPU Sovereignty (GPU主権)

> VRAM は計算専用の聖域。計算に直結しないデータの持ち込みを禁ずる。

### 2. CPU Staging (CPUステージング)

> チェックポイントロードは必ず `map_location='cpu'` 経由。
> 必要な重みのみを選択的にGPUへ転送。

### 3. Clean Room Protocol

> ELO評価は VRAMSanitizer によって保護された Clean Room 内で実行。
> 学習用リソースを解放 → 評価 → 再構築。

### 4. Caller Responsibility

> 呼び出し元が自分のリソースを片付けてから、評価を依頼する。

---

## 実装成果物

### 新規作成

| ファイル | バージョン | 内容 |
|----------|-----------|------|
| `elo.py` | v1.3.0 | ELO評価システム、VRAMSanitizer |
| `long_training_v4.py` | v4.1.0 | 並列化学習、Clean Room Protocol |
| `monitor.sh` | v1.0 | Discord通知 + 監視スクリプト |

### 主要クラス

- `VRAMSanitizer`: VRAM浄化のContext Manager
- `ELOEvaluator`: GPU-Native な対戦評価
- `CheckpointManager`: CPU Staging によるモデルロード
- `ParallelTrainer`: 並列化学習ループ

---

## チェックポイント保存のACD対応

| 特性 | 実装 |
|------|------|
| **A (Atomicity)** | 一時ファイル → `shutil.move()` でアトミックリネーム |
| **C (Consistency)** | 保存後に `torch.load()` で検証 |
| **D (Durability)** | `os.fsync()` でディスク永続化 |

---

## 学習曲線

```
Game     1,000: Loss 5.89, ELO 1500
Game    10,000: Loss 5.50, ELO 1502
Game    50,000: Loss 3.80, ELO 1505
Game   100,000: Loss 1.53, ELO 1512
```

---

## ELO評価の制約（Phase II）

| 項目 | 状態 |
|------|------|
| 対戦相手 | 過去の自分のみ |
| 勝敗判定 | 石数差（簡易版） |
| 連の捕獲 | 単石のみ |
| 地の計算 | なし |

**結論:** ELO 1512 は「過去の自分より強い」ことの証明。絶対的な棋力はPhase IIIで検証。

---

## 学んだこと

### 技術的学び

1. **症状ではなく原因を探す**
   - バッチサイズを下げるのは対症療法
   - メモリライフサイクルを追跡して真犯人を特定

2. **設計思想を内面化する**
   - `map_location='cpu'` は知識として知っていた
   - Tessera の GPU Sovereignty として使えるようになった

3. **Tensor の参照を追跡する**
   - Python list に Tensor を溜め続けるとリークする
   - forward hook は明示的に解除が必要

### プロセス的学び

1. **仲間との協力**
   - Claude、Gemini、Copilot の多角的視点
   - 「MambaStateCapture が犯人」は協力で特定

2. **ドキュメント駆動**
   - DESIGN_SPEC を読み返して設計思想を確認
   - Architecture Note として言語化

---

## 次のステップ

### Phase II-c

- [ ] BATCH_SIZE=128 の検証
- [ ] VRAM使用量のモニタリング強化
- [ ] 学習曲線の可視化（Streamlit）

### Phase III

- [ ] 連の完全実装（Connected Components）
- [ ] 地の計算
- [ ] 外部エンジン（GnuGo/KataGo）との対戦
- [ ] 真のELO測定

---

## ファイル一覧

### ログ

- `logs/training_v4_20260113_031019.log` - 学習ログ（317KB）
- `logs/elo_20260113_031023.jsonl` - ELO対戦記録
- `logs/elo_tiles_20260113_031023.jsonl` - タイル毎ELO

### チェックポイント

- `checkpoints/tessera_v4_final_game100000_elo1512.pth` - 最終モデル
- その他約200個の中間チェックポイント

---

## 引用

> "Le symbole donne à penser." — Paul Ricœur

---

*Tessera Project - Phase II-b Complete*
*2026-01-13*
