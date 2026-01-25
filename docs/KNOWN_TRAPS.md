# Tessera Known Traps

Tesseraプロジェクトで発見された技術的な罠（失敗パターン）を記録する。
将来の開発者（人間・AI）が同じ失敗を繰り返さないための知見集。

---

## TRAP-001: パス連打ハック（Pass Spamming Hack）

**発見日:** 2026-01-18
**深刻度:** 🔴 Critical
**状態:** 🔄 対処中

### 症状

- Loss が急激に改善（5.9 → 3.58）
- しかし Win Rate は 0%
- 対局を観察すると、パスを連打してゲームが即終了

### 原因

1. **パス制限なし**: 序盤からパスが可能
2. **終局即報酬**: パス連打で即座に勝敗が決まり、報酬を得られる
3. **サンプリング偏り**: 終局直前のサンプルに偏り、盤面中盤を学習しない

### なぜ危険か

- Loss という指標が「学習成功」を示さない
- 見かけ上の「相転移」が実は「ハックの学習」
- 長時間学習後に発覚すると、時間が無駄になる

### 対策

| 対策 | 説明 |
|------|------|
| **パス制限** | N手（例: 50手）まではパス禁止 |
| **8点サンプリング** | ゲーム全体から均等にサンプル |
| **パスペナルティ** | パス選択に対するペナルティ |
| **定期的なWin Rate確認** | Loss だけでなく eval_quick.py で挙動確認 |

### 教訓

> **「Loss の低下を無批判に喜ばない」**

Loss が改善しても、実際の性能（Win Rate）が伴わなければ意味がない。

---

## TRAP-002: vocab_size の次元混乱

**発見日:** 2026-01-19
**深刻度:** 🟡 Medium
**状態:** ✅ 解決

### 症状

- 文書によって vocab_size の記載が異なる（362, 363, 364）
- 実装と文書の乖離
- 新規参加者が混乱

### 原因

1. **設計変更の未反映**: EOS_TOKEN を廃止したが文書が古いまま
2. **複数の「次元」**: Embedding次元とPolicyHead出力次元が異なる
3. **コミット漏れ**: ローカル変更がGitHubに反映されていない

### 正しい理解

| 数値 | 意味 | 用途 |
|------|------|------|
| 361 | 盤上座標（19×19） | - |
| 362 | 盤上 + PASS | PolicyHead出力（着手可能空間） |
| 363 | 盤上 + PASS + PAD | vocab_size（Embedding次元） |
| 364 | 旧仕様（EOS含む） | **廃止** |

### 対策

1. **トークン設計セクションを全文書に追加**
2. **両方の次元を明記**（vocab_size と PolicyHead出力）
3. **Git同期の徹底**

### 教訓

> **「実装と文書を同時に更新する」**

---

## TRAP-003: MambaStateCapture によるメモリリーク

**発見日:** 2026-01-14（Phase II）
**深刻度:** 🔴 Critical
**状態:** ✅ 解決

### 症状

- ELO評価時にOOM（Out of Memory）が頻発
- 長時間学習で徐々にメモリ使用量が増加

### 原因

- forward hook が hidden state を保持し続けた
- ガベージコレクションが効かない

### 対策

**MambaStateCapture クラスを完全削除**

### 結果

790回のELO評価でOOMゼロ

### 教訓

> **「forward hook は危険。使う場合は必ずメモリ管理を確認」**

---

## TRAP-004: Pythonループによる速度低下

**発見日:** 2026-01-16（Phase III.1）
**深刻度:** 🟡 Medium
**状態:** ✅ 解決

### 症状

- 学習速度が 3.1 g/s で頭打ち
- GPU使用率が低い

### 原因

- 連検出・捕獲処理がPythonループで実装されていた
- CPUとGPU間の同期待ちが発生

### 対策

1. **Tromp-Taylor採用**: 捕獲ロジック自体を廃止
2. **全操作をテンソル演算に**: No Python Loop

### 結果

学習速度 3.1 → 15.6 g/s（5倍向上）

### 教訓

> **「GPU sovereignty を守れ。Pythonループは敵」**

---

## TRAP-005: 文書間の整合性崩壊

**発見日:** 2026-01-19
**深刻度:** 🟡 Medium
**状態:** ✅ 解決

### 症状

- README, HANDOFF, DESIGN_SPEC で記載が矛盾
- Phase III.2 が「完了」なのか「進行中」なのか不明
- どの文書が正しいのかわからない

### 原因

1. **更新漏れ**: 一部の文書だけ更新
2. **責務の曖昧さ**: どの文書に何を書くか不明確
3. **Git同期不足**: ローカル変更がpushされていない

### 対策

1. **文書責務の明確化**: DECISION_LOG, KNOWN_TRAPS 新設
2. **一括更新**: 関連文書を同時に更新
3. **Gitワークフロー徹底**: 変更したら即push

### 教訓

> **「文書は生き物。放置すると腐る」**

---

## TRAP-006: 「同意点即実行」の罠

**発見日:** 2026-01-19
**深刻度:** 🟡 Medium（LLM協業特有）
**状態:** ⚠️ 継続的注意

### 症状

- 複数のAI（Claude, Gemini, Copilot）が同意した内容を即実行
- 後から「実は見落としがあった」と判明

### 原因

- LLMは「同意」しやすい傾向がある
- 反論がないことは正しさの証明ではない
- 全員が同じ盲点を持っている可能性

### 対策

1. **「なぜ同意するのか」を問う**
2. **意図的に反論を求める**
3. **事実（データ、実験結果）で検証**
4. **Decision Log に記録し、後から検証可能に**

### 教訓

> **「同意は安心ではない。検証せよ」**

---

## テンプレート

新しい罠を記録する際は以下のフォーマットを使用：
```markdown
## TRAP-XXX: [罠の名前]

**発見日:** YYYY-MM-DD
**深刻度:** 🔴 Critical / 🟡 Medium / 🟢 Low
**状態:** ✅ 解決 / 🔄 対処中 / ⚠️ 継続的注意

### 症状
[どのような問題が発生したか]

### 原因
[なぜ発生したか]

### 対策
[どう解決したか / 解決するか]

### 教訓
> **「一言でまとめた教訓」**
```

---

*"Le symbole donne à penser."* — Paul Ricœur

## TRAP-007: 文書ベースの推測による実装状態の誤認

**発見日:** 2026-01-19
**深刻度:** 🟡 Medium
**状態:** ✅ 解決

### 症状

- ARCHITECTURE_OVERVIEW作成時、RayCast/Diffusion/Fusionを「将来構想（未実装）」と記載
- 実際にはソースコードに実装済み・有効化済みだった
- 文書（DESIGN_SPEC, HANDOFF等）に記載がなかったため、存在しないと判断

### 原因

1. **文書依存の調査**: LLMは文書を「真実のソース」として扱いがち
2. **ソースコード未確認**: 文書に書いてあることだけを信じた
3. **暗黙知の非文書化**: 開発者（山田さん）は知っていたが、LLMに伝える必要性を感じなかった

### 経緯

- 2025/1/14-15: Diffusion + RayCast + Fusion を実装（Phase III.1）
- 2025/1/17: Tromp-Taylor への方針転換、文書の焦点がそちらに移る
- 2026/1/19: ARCHITECTURE_OVERVIEW作成時に「未実装」と誤記
- 2026/1/19: ソースコード直接調査で実装済みと判明

### 対策

1. **ソースコード直接検査を必須化**: 文書だけでなく `grep` で実装確認
2. **「存在しない」と書く前に検証**: 特に複雑なコンポーネントについて
3. **実装ファイル一覧を文書に含める**: 何が存在するかを明示

### 教訓

> **「文書は不完全。ソースコードこそが真実のソース（Source of Truth）」**

---


## TRAP-008: 視点破綻（Perspective Collapse）

**発見日:** 2026-01-19
**深刻度:** 🔴 Critical
**状態:** ✅ 解決（DEC-008）

### 症状

- Loss が 5.9 で停滞し、相転移が起きない
- Win Rate が 0%（ランダムにすら勝てない）
- しかしパス連打ではなく、石は正常に配置されている
- 評価スクリプトは正常（Random vs Random で komi=0 なら約50%）

### 原因

学習時に盤面を常に「黒=+1, 白=-1」の絶対座標でモデルに渡していた。

白番のとき：
- Reward は winner * perspective で正しく変換されていた
- しかし Board は黒視点固定
- モデルは「自分の石が -1」という矛盾した状態で学習
- 結果、何も学習できず Loss が一様分布（5.88）付近で停滞

### 対策

perspective = 1.0 if idx % 2 == 0 else -1.0
current_board = current_board * perspective

これにより Board / Seq / Reward が整合する。

### 教訓

> **「Reward だけでなく、全ての入力が視点と整合しているか確認せよ」**

## TRAP-009: 盤面と履歴の視点不整合（Perspective Mismatch）

**発見日:** 2026-01-20
**深刻度:** 🔴 Critical
**状態:** ✅ 解決（DEC-009）

### 症状

- DEC-008（視点正規化）を適用したにもかかわらず Win Rate 0% が継続
- Loss(P) が 5.9 付近で停滞し、相転移が起きない
- Pass は 0.2% と正常（パス連打ではない）
- Value Loss もほぼ改善しない

### 原因

**「盤面は主観化したが、履歴は客観のまま」という認知的不整合**

| 入力 | 視点 | 問題 |
|------|------|------|
| 盤面（Board） | 主観 | DEC-008 で「自分=+1, 相手=-1」に正規化済み ✅ |
| 履歴（Sequence） | 客観 | 「誰が打ったか」の情報がない ❌ |

Mamba の入力設計を確認したところ：
```python
# model.py - MambaModel.forward
h = self.embedding(x)  # x は token ID（0〜362）のみ
```

seq には手番情報が一切含まれず、Mamba は「誰の手か」を識別できない状態だった。

**Copilot の表現:**
> 「視覚（boards）と記憶（seq）が別の世界線を見ている」

**Gemini の表現:**
> 「主語（誰が）も時制（いつ）も奪われた『単なる単語（座標）の羅列』を放り込んでいる」

### なぜ危険か

- TRAP-008（視点破綻）を修正しても、この問題が残っていると効果がない
- 盤面だけ主観化しても、履歴が客観のままでは整合性が取れない
- Policy Head が「誰の意志で打てばいいか」を判断できない
- Loss(P) が一様分布（5.88）付近で停滞する

### 対策

**DEC-009: Turn Embedding の導入**
```python
# MambaModel.__init__ 内
self.turn_emb = nn.Embedding(2, config.D_MODEL)  # 0=Self, 1=Other

# MambaModel.forward 内
h = self.embedding(seq)           # [B, T, D] (空間情報)
h = h + self.turn_emb(turn_seq)   # [B, T, D] (主観コンテキストの付与)
```

これにより：
- 各着手が「自分の手か相手の手か」を識別可能に
- 盤面（主観）と履歴（主観化）の整合性が確立
- vocab_size=363 を維持（DEC-002 との整合）
- Information Continuity を維持（Tessera 哲学との整合）

### TRAP-008 との関係

| TRAP | 問題 | 影響範囲 | 解決 |
|------|------|----------|------|
| TRAP-008 | 盤面が黒視点固定 | Board 入力 | DEC-008 |
| TRAP-009 | 履歴に手番情報なし | Sequence 入力 | DEC-009 |

TRAP-008 と TRAP-009 は**両方とも解決しないと Win Rate は改善しない**。

### 教訓

> **「視点の整合性は、全ての入力経路で確保せよ」**

盤面だけ、履歴だけ、ではなく、モデルに渡す**全ての入力**が同じ視点で整合している必要がある。

### 発見の経緯

1. DEC-008 適用後も Win Rate 0% → 「まだ何か漏れている」
2. Copilot が step() の視点変換漏れを指摘 → 修正
3. それでも Win Rate 0% → 「もっと深い問題がある」
4. Gemini が「seq に手番情報がない」と指摘
5. Copilot が「盤面は主観、履歴は客観」という構造的問題を言語化
6. Claude が Tessera 思想との相性を評価表で整理
7. 三者一致で Option B（Turn Embedding）を採用

## TRAP-010: Actor-Critic Divergence（負の強化による発散）

**発見日:** 2026-01-22
**深刻度:** 🔴 Critical
**状態:** ✅ 解決（v0.3.0）

### 症状

- PG Loss が負の無限大に発散
- CE Loss が ln(362)≈5.89 を超えて 23+ に達する
- Win Rate 0% のまま

### 原因

負の Advantage による「確率0への爆発」。

- `pg_loss = CE × Advantage` で、Advantage が負の場合、Loss は CE を最大化しようとする
- クロスエントロピーの勾配 `∂/∂p(-ln p) = -1/p` は `p→0` で無限大に爆発
- 「勝った手を伸ばす力」より「負けた手を罰する力」が圧倒的に強い
- モデルは「全確率を0にしたい」という不可能な解に向かって暴走

### 対策

**Positive Reinforcement Only**: 負の Advantage を学習対象から除外
```python
if self.config.POSITIVE_ADV_ONLY:
    valid_adv_mask = (advantages > 0).float()
    final_weights = weights * valid_adv_mask
```

### 教訓

> **「強化学習で負の報酬を使う場合、勾配の非対称性に注意せよ」**

---

## TRAP-011: Off-by-One Error in Batch Indexing（バッチ化時のインデックスズレ）

**発見日:** 2026-01-22
**深刻度:** 🔴 Critical
**状態:** ✅ 解決（v0.3.1）

### 症状

- 発散は止まったが Win Rate 0% が継続
- CE Loss は正常範囲（5.0付近）
- 学習は進んでいるように見えるが、性能が向上しない

### 原因

`replay_history_to_boards_fast` の仕様誤解によるインデックスズレ。

- `boards[t]` は「moves[t]を打つ**前**」の状態を返す
- コードでは `board_indices = sample_indices - 1` としていた
- これにより「1手前の盤面」を見て予測させていた
- **相手の最後の手が見えない状態で、次の手を予測させていた**

### 検証方法

`debug_batch_semantics.py` で盤面を可視化：
```bash
docker compose exec tessera python3 /app/src/debug_batch_semantics.py
```

Move 4 を予測する際に4つの石が見えるべきところ、3つしか見えなかった。

### 対策
```python
# 修正前（誤り）
board_indices = sample_indices - 1

# 修正後（正しい）
board_indices = sample_indices
```

### 教訓

> **「形状が合っていても、中身（意味）がズレていれば学習は進まない。Semantic Sanity Check を習慣化せよ」**


---

## TRAP-012: The Mamba Spike（局所的勾配爆発）

**発見日**: 2026-01-24
**発見者**: Gemini, Copilot

### 症状

- 学習全体は安定しているのに、突然Grad Maxが100〜200に跳ね上がり自動停止
- 全体のLossは正常範囲でも、特定の層（mamba_layers.x_proj）の勾配ノルムだけが突出

### 原因

- Mambaのx_proj（Input Projection）層は、入力データの外れ値（極端なAdvantageや予期せぬ盤面パターン）に対して敏感
- PyTorchのclip_grad_norm_（Global Clip）は全層を一律に縮小するため、1つの層だけが突出している場合、健全な層の学習まで阻害する

### 対策

1. **Surgical Gradient Scaling**: Global Clipの前に、問題の層だけを個別にクリップ
2. **Data Hygiene**: AdvantageやLoss自体に上限（Cap）を設け、入力値の爆発を防ぐ

### 教訓

- Mamba/Transformers混成モデルでは、**層ごとの勾配ノルムを監視**することが不可欠
- Global Normだけでは問題の所在が見えない
- 「全体が正常でも、局所が暴走する」ケースに注意

---

## TRAP-013: Resume Without Policy（運用方針なきResume）

**発見日**: 2026-01-24
**発見者**: Gemini, Copilot, Claude（十六代目）

### 症状

- チェックポイントからResumeした直後に勾配発散
- weights-only、LR低下、AMP無効化、クリップ強化を試みても回復不能
- 複数のemergency checkpointが生成される

### 原因

- Resume時のデフォルトがfull state（Optimizer状態含む）だった
- チェックポイントの健全性チェック（Norm分布、Param Groups等）がなかった
- 「とりあえず再開」という属人的判断で進めてしまった
- 実装方針（How）はあったが、運用方針（Policy）がなかった

### 対策

1. **Resume Policy制定**: デフォルトをweights-onlyに。full state継承には明示的承認と理由を必須化
2. **safe_start**: 起動前チェックリストを強制するラッパー
3. **Safety Mode**: Resume後Nゲームは低LR・強クリップを自動適用
4. **Kill Switch**: 閾値超過で自動停止する監視デーモン

### 教訓

- 「実装」だけでなく「運用」も明文化せよ
- Parking Lotの安全策を「いつかやる」で後回しにするな
- 「急がば回れ」—基盤を固めてから進め
- 貴重な計算リソースと時間を、確認漏れで失うな
