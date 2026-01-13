# Changelog

All notable changes to Tessera (MambaGo) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Planned
- Phase III: ルールエンジン改善（連の完全実装、終局判定）
- 19^4 テンソル表現の理論的検証

---

## [0.3.0] - 2026-01-14

### Phase II-c 完了: 200,000 ゲーム学習

#### 実績
- **総ゲーム数:** 200,096
- **総手数:** 34,779,675
- **学習時間:** 2.17 時間
- **最終 ELO:** 1496
- **最高 ELO:** 1517
- **Loss 推移:** 5.89（理論値）→ 5.4（ピーク）→ 2.3（最低）→ 4.0（最終）

#### Added
- `long_training_v4.py`: 長期学習スクリプト（ELO 評価、Tile チェックポイント）
- `streamlit_dashboard.py`: リアルタイム学習可視化ダッシュボード v1.2
- BATCH_SIZE 自動スケール（VRAM に応じて 32/64/128 を選択）
- JST タイムゾーン対応

#### Fixed
- **MambaStateCapture 削除による OOM 解決**
  - 原因: forward hook が hidden state を保持し、メモリリーク
  - 解決: MambaStateCapture を完全削除、Mamba 内部で状態管理
  - 効果: ELO 評価 789 回連続成功（OOM ゼロ）
- `losses_window` のチェックポイント永続化
- VRAMSanitizer によるメモリ管理

#### Changed
- BATCH_SIZE: 64 → 128（8GB VRAM 環境）
- NUM_GAMES: 100,000 → 200,000

#### Technical Notes
- Loss の「相転移」現象を観測（175k 付近で急降下）
- 自己対戦の性質上、Loss は収束後も変動する（正常な挙動）
- `best_loss: 0.0000` は再開時の初期化問題（表示のみ、学習に影響なし）

---

## [0.2.0] - 2026-01-12

### Phase II-a/b 完了: 基盤構築と初期学習

#### Added
- GPUGoEngine: バッチ化された盤面管理
- MambaModel: 4層 SSM、1.9M パラメータ
- TesseraMonitor: VRAM、SSM 状態監視
- 自己対局 + 学習ループ

#### Achieved
- 100,000 ゲーム学習完了
- ELO 1512 達成
- GPU 内完結アーキテクチャ確立

---

## [0.1.0] - 2026-01-11

### Phase I 完了: 環境構築

#### Added
- Docker + CUDA 12.6 + mamba-ssm 環境
- WSL2 + RTX 4070 Laptop (8GB) 対応
- 基本的なプロジェクト構造

---

## 設計原則

| 原則 | 説明 |
|------|------|
| GPU Complete | 全操作が GPU 内で完結、CPU 転送ゼロ |
| Batch First | 全操作がバッチ化されたテンソル演算 |
| No Python Loop | ループは torch 演算に置換 |
| Clean Room | 外部棋譜を使用しない、自己対戦のみ |
| Observable | 全ての挙動がモニター可能 |

---

*"Le symbole donne à penser."* — Paul Ricœur
