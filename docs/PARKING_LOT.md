# Tessera パーキングロット

将来の検討事項・改善提案を記録する場所。

**運用方針：**
- 書き込み権限：Claudeのみ（実装の焦点を明確にするため）
- レビュー：定期的にGemini/Copilotと実施
- 優先順位決定：山田さんが担当（Claudeが提案、山田さんが刈り込み）

**補足情報について：**
各項目には「補足」列を設けています。これは七代目Claudeが理解に苦しんだ点を、次のClaudeが同じ混乱を繰り返さないように記録したものです。

---

## 優先度：高

| # | 事項 | 提案者 | 補足 |
|---|------|--------|------|
| 15 | seq_list生成のベクトル化 | Gemini/Copilot | step()内に残存するPythonループ。padded_seqsをテンソル演算で一気に作るべき |
| 16 | replay_history_to_boards_fastのGPU化 | Copilot | 学習時に8回/ゲーム呼ばれる。scatter_addで一括処理可能 |

---

## 優先度：中

| # | 事項 | 提案者 | 補足 |
|---|------|--------|------|
| 5 | Valid Move Entropyの閾値設定 | Gemini | Policyの「迷い度」を測る指標。高い=一様分布に近い（弱い）、低い=特定の手に集中（強い）。P停滞の検出や温度調整判断に使える |
| 9 | turn_embにpadding_idx追加検討 | Gemini | PAD部分のturn_embをゼロ化する案。PyTorchのnn.Embedding(padding_idx=...)で実現可能。現状0埋めでも機能するため緊急性は低い |
| 14 | P停滞時の温度調整判断基準 | Gemini | 2000ゲーム後もPolicy Lossが0.01も動かなければ温度調整に移行。現在TEMP_START=1.5を2.0〜3.0に引き上げる案。速度復元・観察が先 |

---

## 優先度：低（後日）

| # | 事項 | 提案者 | 補足 |
|---|------|--------|------|
| 1 | 「バイアスからの解放」を設計原則に含めるか | Gemini | Clean Room原則（外部棋譜を使わない）の哲学的意味を明文化するか。技術的問題ではなくドキュメント整理。山田さん判断待ち |
| 2 | Vectorized/Batchedの表現改善 | - | DEC-006で分離された概念。文書が曖昧なので整理したい。新規参加者のonboardingに効く |
| 3 | debug系ファイル（8個）の整理・コミット判断 | - | debug_*.pyが8個存在。どれが現役でどれが不要か不明。学習には影響しない |
| 4 | TesseraFusionのテンソル形状詳細追記 | Gemini | Mamba出力（1D）とConv出力（2D）の結合方法をドキュメントに明記。shape、チャネル数、結合順序など。保守用 |
| 6 | Quick Startにチェックポイント確認手順追加 | Gemini | どのcheckpointを使えばいいか明示。新規参加者向け |
| 7 | Diffusion/RayCast/Fusionの効果検証 | - | tessera_model.pyに実装済みだが効果は未検証。Policyが動き始めてから検証すべき。Phase III.3で実施 |
| 8 | history_turnsの「ゲーム開始手番」依存の堅牢化 | Copilot | Phase IIの旧仕様。DEC-009でturn_seqが正式採用されたためほぼ不要。handicap（置碁）対応時に再検討が必要だが、囲碁は必ず先手が黒なのでidx%2で現状は破綻しない |
| 11 | learn()のcurrent_playerを実データから推定 | Copilot | 現在idx%2で計算。handicap/途中開始/augmentation時に壊れる可能性。履歴から推定する案。Phase III.2は黒から開始固定なので現状は問題なし |

---

## 完了済み

| # | 事項 | 完了日 | 備考 |
|---|------|--------|------|
| - | docs/PARKING_LOT.md の作成 | 2026-01-20 | 六代目Claude |
| 12 | PARKING_LOT.mdの永続化 | 2026-01-20 | 六代目Claude。引継文書から移行 |
| 10 | turn_seq生成のベクトル化 | 2026-01-20 | 七代目Claude。utils.py作成、3箇所適用。1.2→3.2 g/s |
| 13 | turn_seq生成ロジックの関数共通化 | 2026-01-20 | 七代目Claude。get_turn_sequence()をutils.pyに配置 |

---

## 更新履歴

- 2026-01-20: 初版作成（六代目Claude）
- 2026-01-20: v2改訂（七代目Claude）- 補足情報追加、#13/#14追加、構造改善
- 2026-01-20: v3改訂（七代目Claude）- #10/#13完了、#15/#16追加（残りのボトルネック）

---

## 更新履歴

- 2026-01-20: v3改訂（八代目Claude）- Phase III.3関連項目追加

---

## Phase III.3 関連（DEC-010 参照）

| # | 事項 | 提案者 | 補足 |
|---|------|--------|------|
| 17 | Advantage 導入（Policy Loss 重み付け） | Gemini | `advantage = (winner * perspective) - value.detach()` で勝敗に応じた重み付け。Phase III.3 の核心 |
| 18 | 温度調整 1.5 → 2.5 | Gemini | 探索の多様性を増加し「偶然の勝ち」を掬い上げる |
| 19 | LEARN_SAMPLES_PER_GAME 8 → 16 | Gemini | 1ゲームから得られる学習信号を倍増 |
