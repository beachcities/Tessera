# Tessera (MambaGo) Phase III 設計仕様書

**Version:** 0.1.0 (Draft)
**Date:** 2026-01-14
**Status:** 設計中
**Author:** beachcities + Claude

---

## 0. Phase III 目標

Phase II で確立した GPU-Native 自己対戦学習基盤の上に、以下を実現する：

| # | 目標 | 定量基準 |
|---|------|----------|
| 1 | ルールエンジンの完全化 | 連の捕獲、完全なコウ判定 |
| 2 | 終局判定の改善 | 地の計算が正確 |
| 3 | 学習メトリクスの改善 | best_loss を移動平均に変更 |
| 4 | 外部評価の準備 | GnuGo/KataGo との対戦インターフェース |

---

## 1. ルールエンジン改善

### 1.1 連の GPU 実装（Connected Components）

**現状:** 単石の捕獲のみ
**目標:** 任意サイズの連を正しく捕獲

**アルゴリズム候補:**
1. Union-Find の GPU 実装
2. 反復的なラベル伝播
3. torch.scatter を使った隣接更新

**設計原則:** Python ループなし、バッチ処理

### 1.2 コウの完全実装

**現状:** 直前のコウのみ判定
**目標:** スーパーコウ対応

**実装方針:**
- 盤面ハッシュの履歴を保持
- GPU 内でハッシュ計算

### 1.3 自殺手判定の完全化

**現状:** 単石の自殺手のみ判定
**目標:** 連を含む自殺手を正確に判定

---

## 2. 終局判定の改善

### 2.1 地の計算

**Phase II:** 二連続パスで終局
**Phase III:** 地の計算を追加

**実装案:**
1. Flood-fill で空点の領域を特定
2. 領域が単一色に囲まれていれば地
3. 係争地（セキなど）の判定

### 2.2 確定領域マスク（検討中）

**アイデア:**
- 確定した地をマスクして計算から除外
- 終盤の計算効率向上

```python
# 概念コード
confirmed_mask = detect_confirmed_territory(board)
active_region = board * (~confirmed_mask)
output = mamba(active_region)
```

**状態:** アイデア段階、実装未定

---

## 3. 学習メトリクスの改善

### 3.1 best_loss の修正

**問題:** 単一ゲームの最小 Loss を記録 → 外れ値に弱い

**解決:**
```python
# 現状
if loss_val < self.stats['best_loss']:
    self.stats['best_loss'] = loss_val

# 修正案
recent_loss = sum(losses_window) / len(losses_window)
if recent_loss < self.stats['best_avg_loss']:
    self.stats['best_avg_loss'] = recent_loss
```

### 3.2 losses_window の永続化

**Phase II-c で実装済み:**
- チェックポイントに `losses_window` を保存
- 再開時に復元

---

## 4. 外部評価インターフェース

### 4.1 目的

自己対戦 ELO は相対的な指標。外部エンジンとの対戦で絶対的な強さを測定。

### 4.2 使用方法

| 用途 | エンジン | 理由 |
|------|----------|------|
| **学習** | 自己対戦のみ | ライセンスクリーン |
| **評価** | GnuGo / KataGo | ELO 測定専用 |

**重要:** 外部エンジンの手を学習データに使用しない（ライセンス考慮）

### 4.3 GTP インターフェース

```python
class GTPInterface:
    """Go Text Protocol でエンジンと通信"""
    def send_move(self, move: str) -> None: ...
    def get_move(self) -> str: ...
    def setup_board(self, sgf: str) -> None: ...
```

---

## 5. 将来検討事項（未検証）

### 5.1 Tesseract Net (19^4 空間リンク)

**起源:** Colab + A100 環境での実験構想（Gemini との協働）

**概念:**
- 盤面 19×19 = 361 地点
- 時間軸 19×19 = 361 手
- 合計: 19^4 = 130,321 の4次元テンソル

```
現状: 361×361 のフラット表現
構想: 19×19×19×19 の4次元テンソル
```

**仮説:**
- Mamba の SSM 特性と相性が良い可能性
- 対角線的なベクトル距離を活かせる
- 361×361 より計算効率が高い可能性

**spatial_link の概念:**
- 盤面の全 361 地点から別の全 361 地点への「潜在的な影響力のリンク」を学習
- 「一軒トビ」「カカリ」などの囲碁の幾何学的概念を重みとして記憶

**状態:**
- [ ] 理論的検証（未実施）
- [ ] 計算量・メモリ使用量の見積もり（未実施）
- [ ] プロトタイプ実装（未実施）

**前提条件:**
- より大きな VRAM 環境（A100 クラス）が必要な可能性
- 現在の RTX 4070 (8GB) では検証困難かもしれない

### 5.2 Incremental Inference

**概念:** 1手ずつ状態更新、全履歴再計算を回避

**状態:** Phase II で見送り、Phase III 以降で検討

### 5.3 Landmark Memory

**概念:** 重要イベント（コウ、大石死）のスパース保持

**状態:** アイデア段階

---

## 6. ファイル構成（予定）

```
src/
├── gpu_go_engine.py      # 連の完全実装を追加
├── model.py              # 変更なし
├── monitor.py            # 変更なし
├── gtp_interface.py      # 新規: 外部エンジン通信
├── territory.py          # 新規: 地の計算
└── long_training_v5.py   # メトリクス改善版
```

---

## 7. 実装ロードマップ

### Phase III-a: ルールエンジン
| Step | 内容 | 完了条件 |
|------|------|----------|
| 1 | 連の検出（GPU） | バッチ処理で正確に検出 |
| 2 | 連の捕獲 | 呼吸点ゼロで正しく除去 |
| 3 | 自殺手判定 | 連を含めて正確に判定 |

### Phase III-b: 終局と評価
| Step | 内容 | 完了条件 |
|------|------|----------|
| 4 | 地の計算 | 単純な局面で正確 |
| 5 | GTP インターフェース | GnuGo と対戦可能 |
| 6 | ELO 校正 | 外部 ELO との対応付け |

---

## 付録 A: 設計原則（継続）

| 原則 | 説明 |
|------|------|
| GPU Complete | 全操作が GPU 内で完結 |
| Batch First | 全操作がバッチ化 |
| No Python Loop | ループは torch 演算に置換 |
| Clean Room | 外部棋譜を使用しない |
| Observable | 全ての挙動がモニター可能 |

---

## 付録 B: ライセンス考慮事項

**学習データ:**
- 自己対戦のみ使用 → 完全にクリーン

**外部エンジン:**
- GnuGo (GPL v3): 評価専用、学習には使用しない
- KataGo (MIT): 評価専用、学習への使用も法的にはグレー

**著作権法30条の4:**
- AI 学習への著作物利用を広く認める
- ただし「利用」（生成・商用化）は別途検討が必要

**方針:** 安全のため、外部エンジンは評価専用とする

---

*"Le symbole donne à penser."* — Paul Ricœur
