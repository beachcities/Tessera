# Tessera Implementation Principles

**Version:** 1.0
**Date:** 2026-01-21
**Authors:** Copilot (draft), Gemini & Claude Opus (review), 山田政幸 (approval)

---

## 前文

Tessera の実装原則は、
人とAIが本質的に抱える弱さを前提としている。

私たちは形状を取り違え、勾配を切り、デバイスを混在させ、
曖昧さに流され、直感に裏切られ、
「動いているように見えて壊れている」状態を生み出してしまう。

PyTorch は寛容で、曖昧さを許容し、
静かに混乱を忍び込ませる。

だからこそ、私たちは **曖昧さを排除し、整合性を保つための明確な原則** を定める必要がある。

実装原則とは、弱さを否定するためのものではない。
**弱さを受け入れ、その上で秩序を維持するための実務的な原則** である。

---

## I. Contract（境界の宣言）

曖昧さが侵入しないよう、意味・境界・責任を明示的に定義する領域。

- **Time-step Semantics** — 時間軸の意味を明確に定義する
- **Board Encoding Contract** — 盤面エンコーディングの契約を明示する
- **Loss Decomposition Contract** — 損失関数の分解と責任を明確にする
- **RNG Scope Contract** — 乱数生成のスコープと再現性を定義する
- **Statefulness Boundary** — 状態を持つ/持たないの境界を明示する
- **Device Ownership Contract** — デバイス所有権を明確にする
- **Normalization Contract** — 正規化の責任範囲を定義する
- **Logging Contract** — ログ出力の契約を明示する

---

## II. Tensor Integrity（整合性の保持）

テンソルの完全性・一貫性を守り、
形状・勾配・デバイス・再現性の乱れを排除する領域。

- **Deterministic Reproducibility** — 決定論的な再現性を保証する
- **Shape & Semantic Documentation** — 形状と意味をコメントで文書化する
- **Device Integrity** — デバイスの一貫性を保つ
- **Gradient Flow Integrity** — 勾配の流れを意図通りに保つ
- **Active Shape Guarding** — モデル入力やLoss計算などの重要境界（Critical Boundaries）では、`assert tensor.shape == (...)` を挿入し、形状不整合を即座にクラッシュさせる（Fail-Fast）
- **Explicit Broadcasting** — `unsqueeze` や `view` による次元操作を行う直前・直後には、操作の意図と結果形状を明示する。暗黙のブロードキャストに頼らない

---

## III. Data Representation（データ表現）

データの形と意味を揺らさず、
モデルが常に同じ世界を見られるようにする領域。

- **右詰め整形** — シーケンスは右詰めでパディングする
- **PAD/padding_idx の一貫した扱い** — PADトークンの処理を統一する

---

## IV. Training Integrity（学習整合性）

推論と学習の間に矛盾が生まれないよう、
学習過程の一貫性を保証する領域。

- **Replay Consistency** — リプレイ時の盤面再構築が推論時と一致すること
- **Memory Budget Awareness** — VRAMバジェットを意識した設計

---

## V. Debugging & Verification（検証）

弱さが生む誤りを早期に発見し、
破綻が深層に伝播する前に止める領域。

- **Explainability** — 挙動を説明可能にする
- **One-change-at-a-time** — 一度に一つの変更のみ行う
- **Invariant & Fail-Fast Validation** — 不変条件を検証し、早期に失敗させる
- **Test at Boundaries** — 境界条件でテストする

---

## VI. Code Quality（品質）

未来の自分や他のAIが理解できるよう、
コードの意味・意図・責任を明確に残す領域。

- **DRY（共通化）** — 重複を排除し、共通関数に抽出する
- **Atomic Commit** — 一つのコミットは一つの論理的変更
- **Defense Against Future Self** — 将来の自分（や次のAI）が誤解しないようにする
- **No Silent Fallback** — 暗黙のフォールバックを禁止する
- **Configuration as Code** — 設定をコードとして管理する

---

## VII. Performance（性能）

秩序と整合性を守ったうえで、
GPU-native の速度を最大化する領域。

- **GPU-first** — GPUでの実行を最優先に設計する
- **ベクトル化を優先** — Pythonループよりテンソル演算を選ぶ
- **バッチ化を常に検討** — 複数ゲーム/サンプルの同時処理を考える
- **Python ループは最後の手段** — どうしても必要な場合のみ許容
- **preallocation（append禁止）** — リストへのappendではなく事前確保
- **マスク駆動ロジック** — 条件分岐よりマスク演算を選ぶ

---

## 設計原則との関係

この実装原則は、Tessera の設計原則を実装レベルに落とし込んだものである。

| 設計原則 | 関連する実装原則 |
|----------|------------------|
| **GPU Complete** | I. Device Ownership, VII. GPU-first |
| **Vectorized** | VII. ベクトル化を優先, Python ループは最後の手段 |
| **Batched** | VII. バッチ化を常に検討 |
| **Clean Room** | VI. Configuration as Code |
| **Observable** | I. Logging Contract, V. Explainability |

---

*"Le symbole donne à penser."* — Paul Ricœur
