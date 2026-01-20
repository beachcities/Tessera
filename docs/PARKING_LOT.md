# Tessera パーキングロット

将来の検討事項・改善提案を記録する場所。

**運用方針：**
- 書き込み権限：Claudeのみ（実装の焦点を明確にするため）
- レビュー：定期的にGemini/CoPilotと実施
- 優先順位決定：Claudeが担当

---

## 優先度：高

| # | 事項 | 提案者 | 備考 |
|---|------|--------|------|
| 10 | turn_seq生成のベクトル化 | Gemini | 速度改善に直結。arange + mod 2 + broadcast |

---

## 優先度：中

| # | 事項 | 提案者 | 備考 |
|---|------|--------|------|
| 8 | history_turnsの「ゲーム開始手番」依存の堅牢化 | Copilot | handicap対応時に必要 |
| 9 | turn_embにpadding_idx（3番目のID）を追加する検討 | Gemini | PADトークン用 |
| 11 | learn()のcurrent_playerを実データから推定 | Copilot | データ拡張時に必要 |

---

## 優先度：低（後日）

| # | 事項 | 提案者 | 備考 |
|---|------|--------|------|
| 1 | 「バイアスからの解放」を設計原則に含めるか | Gemini | 山田さん判断待ち |
| 2 | Vectorized/Batchedの表現改善 | - | ドキュメント整備 |
| 3 | debug系ファイル（8個）の整理・コミット判断 | - | リポジトリ整理 |
| 4 | TesseraFusionのテンソル形状詳細追記 | Gemini | ドキュメント整備 |
| 5 | Valid Move Entropyの閾値設定 | Gemini | 評価指標 |
| 6 | Quick Startにチェックポイント確認手順追加 | Gemini | ドキュメント整備 |
| 7 | Diffusion/RayCast/Fusionの効果検証 | - | Phase III.3 |

---

## 完了済み

| # | 事項 | 完了日 | 備考 |
|---|------|--------|------|
| - | docs/PARKING_LOT.md の作成 | 2026-01-20 | 六代目Claude |

---

## 更新履歴

- 2026-01-20: 初版作成（六代目Claude）
