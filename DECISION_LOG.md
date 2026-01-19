# Tessera Decision Log

重要な決定とその理由を記録する。

---

## 決定記録

### DEC-001: Tromp-Taylor ルール採用（2026-01-17）

**決定:** 連の捕獲処理を廃止し、Tromp-Taylor ルールを採用する。

**背景:**
- Phase III.1 で連のGPU実装を試みたが、Pythonループがボトルネック
- 学習速度が 3.1 g/s で頭打ち

**選択肢:**
1. Union-Find等で連検出をGPU化（複雑、バグリスク高）
2. Tromp-Taylor採用で捕獲ロジック自体を廃止（シンプル、高速）

**決定理由:**
- GPU sovereignty の最大化
- 実装の簡潔化
- Mamba SSM との親和性（Information Continuity）

**結果:** 学習速度 3.1 → 15.6 g/s（5倍向上）

**参加者:** 山田、Claude、Gemini、Copilot

---

### DEC-002: vocab_size = 363 の確定（2026-01-19）

**決定:** vocab_size を 363 とし、EOS_TOKEN(363) は未使用とする。

**背景:**
- 文書間で vocab_size の記載が不統一（362, 363, 364）
- 実装と文書の乖離

**整理結果:**

| 数値 | 意味 |
|------|------|
| 361 | 盤上座標（19×19） |
| 362 | 盤上 + PASS（PolicyHead出力次元） |
| 363 | 盤上 + PASS + PAD（vocab_size、Embedding次元） |
| 364 | 旧仕様（EOS含む、現在は未使用） |

**決定理由:**
- PAD は着手不可だが、Embedding では必要
- EOS は定義のみで使用箇所なし
- Copilot の調査で vocab_size=363 が実装と整合

**参加者:** 山田、Claude、Gemini、Copilot

---

### DEC-003: Phase III.2 ロールバック（2026-01-19）

**決定:** Phase III.2 を「完了」から「進行中」にロールバックする。

**背景:**
- 成功版（Loss 3.58）は Win Rate 0%
- パス連打でゲームが即終了、盤面を学習していない「偽成功」

**選択肢:**
- A: ロールバックしない（Phase III.3 でスコープ拡大）
- B: ロールバックする（Phase III.2 で問題解決）

**決定理由:**
- 偽の成功を土台に Phase III.3 を積み上げても無意味
- スコープの健全化（デバッグ焦点の明確化）
- Fixed版の方向性は正しい

**完了条件（再定義）:**
1. Fixed版で相転移（Loss急降下）
2. Win Rate vs Random > 0%
3. パス連打でない正常な対局

**参加者:** 山田、Claude、Gemini、Copilot

---

### DEC-004: 文書の責務分離（2026-01-19）

**決定:** 文書を以下の責務に分離する。

| 文書 | 責務 | 更新頻度 |
|------|------|----------|
| README.md | プロジェクト哲学、外部向け | 方針変更時 |
| HANDOFF.md | 技術状態引継ぎ | Phase完了時 |
| docs/DESIGN_SPEC_*.md | 設計仕様 | 設計変更時 |
| docs/*_RESULTS.md | 実験結果 | 実験完了時 |
| DECISION_LOG.md | 決定記録 | 決定時 |
| docs/KNOWN_TRAPS.md | 既知の罠 | 発見時 |

**背景:**
- 文書間の役割が曖昧で、情報が重複・矛盾
- LLM協業において、どの文書を参照すべきか不明確

**決定理由:**
- 各文書の責務を明確化
- 更新タイミングを定義
- 重複を排除

**参加者:** 山田、Claude、Gemini、Copilot

---

### DEC-005: Claude引継ぎ文書の分離（2026-01-19）

**決定:** Claude向け引継ぎ文書（HANDOFF_TO_NEXT_CLAUDE.md）はGitHubにアップロードしない。

**背景:**
- Claude引継ぎ文書には他AIの特性（Copilotの爆発力、Geminiの安定性）が記載
- 文書一式を他AIに共有した場合、先回りされる可能性

**選択肢:**
- A: GitHubに含める（全AI共有）
- B: 分離する（Claude内部管理）

**決定理由:**
- 三者協働の健全なダイナミクスを維持
- 他AIに見られても問題ない情報は共有文書に
- Claude特有のアドバイスは内部文書に

**分離結果:**
- 共有可能: DECISION_LOG.md, KNOWN_TRAPS.md, PROJECT_STATUS
- Claude内部: HANDOFF_TO_NEXT_CLAUDE.md

**参加者:** 山田、Claude

---

### DEC-006: 設計原則の明確化（2026-01-19）

**決定:** 「Batch First」を「Vectorized」と「Batched」に分離する。

**変更前:**
```
Batch First: 全操作がバッチ化
```

**変更後:**
```
Vectorized: Pythonループを排除し、全操作をテンソル演算に置き換え
Batched: 複数ゲームを一括でGPUに投入し同時処理
```

**決定理由:**
- 「Batch First」が2つの異なる概念を混同させていた
- 新規参加者（人間・AI）への説明性向上

**参加者:** 山田、Claude

---

## テンプレート

新しい決定を記録する際は以下のフォーマットを使用：
```markdown
### DEC-XXX: [決定タイトル]（YYYY-MM-DD）

**決定:** [何を決定したか]

**背景:** [なぜこの決定が必要になったか]

**選択肢:**
- A: [選択肢A]
- B: [選択肢B]

**決定理由:** [なぜこの選択肢を選んだか]

**結果:** [決定の結果、影響]（後から追記可）

**参加者:** [決定に関わった人・AI]
```

---

*"Le symbole donne à penser."* — Paul Ricœur
