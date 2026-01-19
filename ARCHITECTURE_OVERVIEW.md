# Tessera Architecture Overview

**Version:** v4
**Date:** 2026-01-19
**Primary Author:** Copilot
**Reviewers:** Gemini, Claude
**Approver:** 山田政幸

---

> **「Tesseraは、囲碁を『計算』するのではなく、GPU上の物理現象として『記述』する試みである。」**

---

## 📖 Reading Order（この文書の位置づけ）

Tessera の文書体系は階層構造になっている。
この文書は「全体地図」として、哲学と実装の間をつなぐ。
```
README.md              — 哲学と入口
    ↓
ARCHITECTURE_OVERVIEW.md（本書） — 全体構造の俯瞰
    ↓
DESIGN_SPEC_PHASE_III.md       — 詳細設計
    ↓
src/                           — 実装
```

### この文書の想定読者

- 新規参加AI（Claude, Gemini, Copilotの次世代）
- 新規参加の人間開発者
- 長期中断後に復帰する山田さん自身

### この文書で得られるもの

- Tessera の全体構造の理解（10分）
- 各コンポーネントの役割と関係性
- 詳細設計・実装への入口

### この文書で得られないもの

- 詳細な実装仕様 → `DESIGN_SPEC_PHASE_III.md`
- 決定の経緯 → `DECISION_LOG.md`
- 失敗パターン → `docs/KNOWN_TRAPS.md`

---

## 🎯 Purpose

この文書は、Tessera のアーキテクチャを **一望できる地図** として提供する。

詳細仕様（DESIGN_SPEC）と実装（src/）の間に橋を架け、
新規参加者（人間・AI）が迷わずに Tessera の内部構造へ入れるようにする。

Tessera は「GPU-native」「Vectorized」「Batched」「Clean Room」という原則の上に構築された、
**Mambaベースの自己対戦型 Go AI** である。

---

## 1. High-Level System Diagram
```
+--------------------------------------------------------------+
|                          Tessera                             |
|--------------------------------------------------------------|
|  Self-Play Loop (Phase III.2)                                |
|    ├── Game Simulator (Tromp–Taylor)                         |
|    ├── Tesseract Field (Diffusion + RayCast)                 |
|    ├── PolicyHead (362-dim logical action space)             |
|    ├── ValueHead (scalar win prob)                           |
|    └── Replay Buffer (batched)                               |
|                                                              |
|  Neural Model (TesseraModel)                                 |
|    ├── Embedding Layer (vocab_size=363)                      |
|    ├── Mamba Blocks (stacked SSM with stateful context)      |
|    ├── DiffusionEngine (物理場生成)                           |
|    ├── RayCastLayerV2 (長距離ポテンシャル)                    |
|    ├── TesseraFusion (Late Fusion)                           |
|    ├── PolicyHead                                            |
|    └── ValueHead                                             |
|                                                              |
|  Training Loop                                               |
|    ├── Vectorized Loss                                       |
|    ├── Batched Backprop                                      |
|    └── torch.compile acceleration                            |
+--------------------------------------------------------------+
```

---

## 2. Core Architectural Principles

### 2.1 GPU Sovereignty

Tessera は CPU に処理を逃がさず、**全工程を GPU 上で完結**させる。

| 処理 | 実装 |
|------|------|
| 盤面更新 | GPU テンソル操作 |
| 地計算（Area Scoring） | Conv2d flood-fill |
| Diffusion | GPU native |
| RayCast | GPU native |
| Mamba forward | GPU native |
| サンプリング | torch.multinomial |
| 損失計算 | Vectorized |

**参照:**
- DEC-001（Tromp–Taylor 採用）
- TRAP-004（Python ループ禁止）

#### Information Continuity（Tromp–Taylor × Mamba の相乗効果）

Tromp–Taylor では**石が盤上から消えない**。

これにより Mamba の SSM State に **一貫した文脈（Information Continuity）** を提供し、
勾配の流れが安定する。石の捕獲による盤面の不連続な変化がないため、
Mamba が「局面の物語」を途切れなく学習できる。

### 2.2 Vectorized / Batched

Tessera の速度と安定性は、ベクトル化されたゲームシミュレーションに依存する。

- N 個のゲームを同時に進行
- N 個の盤面を同時に評価
- N 個の手を同時にサンプリング
- N 個の損失を同時に計算

**参照:**
- DEC-006（Vectorized/Batched 原則の明確化）

### 2.3 Clean Room

Tessera は「Clean Room 原則」を採用する。

- 既存の Go AI 実装を参照しない
- 外部棋譜（プロの対局記録等）を使用しない
- 盤面ルールは Tromp–Taylor をゼロから実装
- 文化的・技術的に自律したアーキテクチャを維持

**理由:**
- ライセンス汚染リスクの回避
- 「人間のバイアスからの解放」という哲学的選択

**参照:** README.md（設計原則）

---

## 3. Model Architecture

### 3.1 Tokenization & Embedding

| ID | 意味 | 備考 |
|----|------|------|
| 0-360 | 盤上座標 (19×19) | |
| 361 | PASS | |
| 362 | PAD | 学習時 ignore_index |
| 363 | EOS | 予約済み、未使用 |

- **物理 vocab_size = 363**
- **論理アクション空間 = 362**（PAD除外）

Embedding は盤面を「言語」として扱うための基盤。

**参照:**
- DEC-002（vocab_size = 363 の確定）
- TRAP-002（次元混乱の罠）

### 3.2 Tesseract Field（物理場としての盤面表現）

**設計思想:**

囲碁の「勢力」「模様」「影響圏」といった概念を、物理的な場として表現する。
従来のCNN/Transformerとは異なる、物理シミュレーション的な特徴抽出を目指す。

**実装状態:** ✅ 実装済み・有効化済み（効果未検証）

**パイプライン:**
```
盤面 [B, 19, 19]
    ↓ DiffusionEngine
物理場スナップショット (phi_b, phi_w) [B, 3, 19, 19] each
    ↓ torch.cat
統合物理場 [B, 6, 19, 19]
    ↓ RayCastLayerV2
長距離ポテンシャル [B, 16, 19, 19]
    ↓ TesseraFusion (with Mamba output)
統合表現 [B, d_model, 19, 19]
```

**関連ファイル（2025/1/14-15 に集中開発）:**

| ファイル | 内容 |
|----------|------|
| `diffusion.py` | DiffusionEngine |
| `ray_index.py` | 8方向インデックステーブル |
| `ray_cast.py` | RayCastLayer 初期版 |
| `ray_cast_v2.py` | RayCastLayerV2（採用版） |
| `fusion_layer.py` | TesseraFusion |

### 3.3 DiffusionEngine

**実装状態:** ✅ 実装済み・有効化済み（効果未検証）

**役割:** 盤面から物理場スナップショットを生成
```python
# tessera_model.py より
self.diffusion = DiffusionEngine(
    steps=diffusion_steps,        # default: 10
    snapshot_steps=snapshot_steps  # default: [2, 5, 10]
)

# forward
phi_b, phi_w, _ = self.diffusion(board)  # [B, 3, 19, 19] each
```

**パラメータ:**

| パラメータ | デフォルト | 意味 |
|------------|------------|------|
| steps | 10 | 拡散ステップ数 |
| snapshot_steps | [2, 5, 10] | スナップショット取得タイミング |

### 3.4 RayCastLayerV2

**実装状態:** ✅ 実装済み・有効化済み（効果未検証）

**役割:** 物理場から8方向の長距離ポテンシャルを計算
```python
# tessera_model.py より
self.raycast = RayCastLayerV2(
    c_in=ray_c_in,           # スナップショット数 × 2 = 6
    c_out=ray_c_out,         # default: 16
    init_scale=ray_init_scale # default: 0.1
)

# forward
field = torch.cat([phi_b, phi_w], dim=1)  # [B, 6, 19, 19]
ray_features = self.raycast(field)         # [B, 16, 19, 19]
```

**パラメータ:**

| パラメータ | デフォルト | 意味 |
|------------|------------|------|
| c_in | 6 | 入力チャンネル（phi_b + phi_w） |
| c_out | 16 | 出力チャンネル |
| init_scale | 0.1 | 重みの初期化スケール |
| distance_decay | 1.0 | 距離減衰（1.0 = 1/d, 2.0 = 1/d²） |

### 3.5 TesseraFusion（Late Fusion）

**実装状態:** ✅ 実装済み・有効化済み（効果未検証）

**役割:** Mamba出力とRayCast出力を統合
```python
# fusion_layer.py より
self.fusion = TesseraFusion(
    mamba_ch=d_model,         # default: 256
    ray_ch=ray_c_out,         # default: 16
    ray_init_scale=0.1        # RayCast側の初期重みを小さく
)

# forward
fused = self.fusion(mamba_features, ray_features)  # [B, d_model, 19, 19]
```

**設計意図:**
- 学習初期は Mamba 優位（ray_init_scale=0.1）
- 学習が進むと RayCast が徐々に効き始める

### 3.6 Mamba Blocks（Stateful Architecture）

Tessera が Transformer ではなく **Mamba** を採用する理由：

| 特性 | Transformer | Mamba |
|------|-------------|-------|
| VRAM効率 | O(n²) | O(n) |
| 長距離依存 | Attention | SSM State |
| 空間的連続性 | 位置エンコーディング依存 | スキャン順序で保持 |

**Mamba の核心的優位性：Stateful Context**

Tesseraは盤面を毎回ゼロから読み直すのではなく、
MambaのStateを一種の **「短期記憶の地層」** として活用する。

これにより、直前の数手が生み出した **「勢い（厚み）」** を、
計算量を抑えたまま次の一手に反映できる。

**設計判断への示唆:**
- 過去数手の履歴をどこまでEmbeddingに入れるべきか？
- → **Stateが持っているから最小限で良い**

### 3.7 PolicyHead

| 項目 | 値 |
|------|-----|
| 物理出力次元 | 363（Embedding と一致） |
| 論理アクション空間 | 362（PAD 除外） |
| 活性化 | Softmax |

- PAD (ID 362) は学習時の `ignore_index` のみで使用
- 推論時はスライス `[:, :362]` により除外

**参照:** TRAP-002（次元混乱の罠）

### 3.8 ValueHead

| 項目 | 値 |
|------|-----|
| 出力 | スカラー [-1, +1] |
| 活性化 | Tanh |
| 意味 | 勝率予測 |

- 自己対戦の安定性に直結
- Phase III.3 で強化予定

---

## 4. Self-Play Architecture

### 4.1 Game Simulator（Tromp–Taylor）

- **GPU 上で完全実装**
- 捕獲は存在しない（石は盤上に残る）
- 地計算は Conv2d flood-fill (`chain_utils.py`)
- 終局判定: 両者連続パス

**注意:** PASS 連打問題は TRAP-001 として管理

### 4.2 Batched Self-Play
```python
# 擬似コード
games: [batch_size, 19, 19]      # N個の盤面
features = TesseractField(games)  # Diffusion + RayCast
logits, value = model(seq, games) # Mamba + Fusion
action = sample(logits)           # サンプリング
games = apply_move(games, action) # 盤面更新
```

1手ごとに N ゲームが同時に進む。

---

## 5. Training Loop

### 5.1 Loss
```
Total Loss = Policy Loss + α × Value Loss + β × Entropy Bonus
```

### 5.2 Acceleration

- `torch.compile` による高速化
- CPU ↔ GPU の往復を禁止
- Python ループ排除

---

## 6. Observability

Tessera は **「観測可能性」** を重視する。

### 標準メトリクス

| メトリクス | 目的 |
|------------|------|
| Win Rate (vs Random) | 基礎性能 |
| Policy Loss | 学習進捗 |
| Value Loss | 形勢判断能力 |
| 対局ログ | 定性的確認 |

### Valid Move Entropy（偽成功防止）

モデルが PASS（361）に確率を集中させていないかを検知するため、
**有効手のエントロピー**を監視する。

| エントロピー | 解釈 |
|--------------|------|
| 低すぎる | PASS 連打の兆候（TRAP-001） |
| 高すぎる | ランダム化（学習不足） |
| 適度 | 健全な探索 |

**参照:** TRAP-001（パス連打ハック）

---

## 7. Phase Alignment

| Phase | 内容 | 状態 |
|-------|------|------|
| II | 基礎アーキテクチャ | ✅ 完了 |
| III.1 | TesseraModel統合（Diffusion/RayCast/Fusion） | ✅ 完了 |
| III.2 | Tromp-Taylor + Value Head | 🔄 進行中（ロールバック） |
| III.3 | 評価と検証 | ⏳ 待機中 |

### Phase III.2 ロールバックの歴史的意義

数値上の Loss 低下（5.9 → 3.58）よりも、
**Win Rate 0%** という実態を優先し、Phase III.2 を再定義した。

これは Tessera の文化である **「誠実さ」と「観測可能性」** を象徴する判断である。

**参照:** DEC-003（Phase III.2 ロールバック）

---

## 7.5 Open Questions（未確定事項）

### 効果未検証

| 項目 | 状態 | 備考 |
|------|------|------|
| Diffusion の効果 | ❓ | Loss/WinRate への寄与不明 |
| RayCast の効果 | ❓ | Loss/WinRate への寄与不明 |
| Fusion の効果 | ❓ | Loss/WinRate への寄与不明 |

※ これらは実装済み・有効化済みだが、効果検証は Phase III.3 で実施予定

### 仮説（未検証）

| 項目 | 提案者 | 備考 |
|------|--------|------|
| ValueHead の最適構造（MLP vs Attention） | - | |
| サンプリング密度（8 / 16 / 32） | Gemini | 相転移加速の可能性 |
| 学習率の再検証 | Gemini | Fixed版で勾配特性が変化 |
| Valid Move Entropy の閾値設定 | Gemini | |
| Diffusion/RayCast バイパスフラグ | Gemini | 効果検証用 |

### 実装済みだが効果未検証

| 項目 | 現状 |
|------|------|
| 8点サンプリング | 相転移への効果は未確認 |
| パス制限（50手） | 妥当性は検証中 |

---

## 8. Future Extensions

- MCTS なしでの強化学習
- ValueHead の強化
- Diffusion/RayCast/Fusion の効果検証
- 大規模バッチによる相転移の促進

---

## 9. Implementation Files

| コンポーネント | ファイル |
|----------------|----------|
| TesseraModel 統合 | `src/tessera_model.py` |
| DiffusionEngine | `src/diffusion.py` |
| RayCastLayerV2 | `src/ray_cast_v2.py` |
| RayIndexBuffer | `src/ray_index.py` |
| TesseraFusion | `src/fusion_layer.py` |
| Game Simulator | `src/gpu_go_engine.py` |
| 地計算 | `src/chain_utils.py` |
| Training Loop | `src/train_phase3_2_fixed.py` |
| Win Rate 評価 | `src/eval_quick.py` |

---

## 10. Quick Start for New Contributors
```bash
# 1. 環境確認
cd ~/GoMamba_Local
git status  # ローカルとGitHubの同期確認

# 2. 現在の状態を把握
cat HANDOFF.md           # 技術状態
cat DECISION_LOG.md      # 決定履歴
cat docs/KNOWN_TRAPS.md  # 避けるべき罠

# 3. Fixed版の学習状態を確認
python src/eval_quick.py  # Win Rate確認

# 4. 学習を継続する場合
python src/train_phase3_2_fixed.py
```

---

## 11. Summary

Tessera のアーキテクチャは、

**GPU-native / Vectorized / Batched / Clean Room**

という原則の上に構築された、**Mamba ベースの自己対戦型 Go AI** である。

**Tesseract Field（Diffusion + RayCast + Fusion）** は、
囲碁の「勢力」「模様」「影響圏」を物理的な場として表現する独自アプローチ。
実装済み・有効化済みだが、効果検証は今後の課題。

Mamba の SSM State が「局面の物語」を保持し、
Tromp-Taylor の Information Continuity がその連続性を保証する。

この文書は、Tessera の内部構造を俯瞰し、
後続の人間・AI が迷わずに貢献できるための **"地図"** として機能する。

---

## Document History

| Version | Date | Authors | Changes |
|---------|------|---------|---------|
| v1 | 2026-01-19 | Copilot | Initial draft |
| v2 | 2026-01-19 | Copilot + Gemini + Claude | Integrated feedback |
| v3 | 2026-01-19 | Claude | Gemini's deep dive + Claude's integrity checks |
| v4 | 2026-01-19 | Claude | RayCast/Diffusion/Fusion を「実装済み・効果未検証」に修正 |

---

*"Le symbole donne à penser."* — Paul Ricœur
