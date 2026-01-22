"""
GPUGoEngine - GPU Native 囲碁エンジン (Tromp-Taylor 版)

Phase III.2: 捕獲処理を廃止、終局時の地計算で勝敗決定

設計思想:
- 全操作がGPU内で完結（CPU転送ゼロ）
- バッチ処理による並列化
- Tromp-Taylor ルール（捕獲なし、終局時Area Scoring）

Version: 0.3.0 (Tromp-Taylor)
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from chain_utils import compute_territory


# ============================================================
# Constants
# ============================================================

BOARD_SIZE = 19
PASS_TOKEN = 361
PAD_TOKEN = 362
EOS_TOKEN = 363
# VOCAB_SIZE は削除: Phase II (364) と Phase III (363) で異なるため、
# エンジン層は物理定数のみを扱い、vocab_size は各モデルで管理する。
# 経緯: DEC-007 参照

# 隣接カーネル（上下左右）
NEIGHBOR_KERNEL = torch.tensor([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=torch.float32).view(1, 1, 3, 3)


# ============================================================
# Utility Functions
# ============================================================

def coord_to_token(row: int, col: int) -> int:
    """座標をトークンに変換"""
    return row * BOARD_SIZE + col


def token_to_coord(token: int) -> Tuple[int, int]:
    """トークンを座標に変換"""
    return token // BOARD_SIZE, token % BOARD_SIZE


def tokens_to_coords_batch(tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """トークンを座標に変換（バッチ対応）"""
    rows = tokens // BOARD_SIZE
    cols = tokens % BOARD_SIZE
    return rows, cols


# ============================================================
# Main Engine Class
# ============================================================

class GPUGoEngine:
    """
    GPU Native 囲碁エンジン (Tromp-Taylor 版)
    
    Tromp-Taylor ルール:
    - 対局中の捕獲なし（石は盤上に残る）
    - 自殺手も有効（ただし空点にのみ着手可能）
    - 終局は両者パスまたは MAX_MOVES
    - 終局時に Area Scoring で勝敗決定
    """
    
    def __init__(self, 
                 batch_size: int, 
                 device: str = 'cuda',
                 max_history: int = 512):
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.max_history = max_history
        self.neighbor_kernel = NEIGHBOR_KERNEL.to(self.device)
        self.reset()
    
    def reset(self):
        """全ゲームを初期状態にリセット"""
        B = self.batch_size
        D = self.device
        
        self.boards = torch.zeros(B, 2, BOARD_SIZE, BOARD_SIZE, device=D)
        self.turn = torch.zeros(B, dtype=torch.long, device=D)
        self.ko_point = torch.full((B,), -1, dtype=torch.long, device=D)
        self.last_move = torch.full((B,), -1, dtype=torch.long, device=D)
        self.consecutive_passes = torch.zeros(B, dtype=torch.long, device=D)
        self.move_count = torch.zeros(B, dtype=torch.long, device=D)
        self.history = torch.full((B, self.max_history), PAD_TOKEN, dtype=torch.long, device=D)
    
    def reset_selected(self, mask: torch.Tensor):
        """選択したバッチのみリセット"""
        B = self.batch_size
        D = self.device
        
        if not mask.any():
            return
            
        self.boards[mask] = 0
        self.turn[mask] = 0
        self.ko_point[mask] = -1
        self.last_move[mask] = -1
        self.consecutive_passes[mask] = 0
        self.move_count[mask] = 0
        self.history[mask] = PAD_TOKEN
    
    # ========================================================
    # Board Queries
    # ========================================================
    
    def get_empty_mask(self) -> torch.Tensor:
        """空点マスクを返す"""
        occupied = self.boards[:, 0] + self.boards[:, 1]
        return occupied == 0
    
    # ========================================================
    # Legal Move Detection
    # ========================================================
    
    def get_legal_mask(self) -> torch.Tensor:
        """
        合法手マスクを返す
        
        Tromp-Taylor: 空点ならどこでも合法（コウ以外）
        """
        B = self.batch_size
        
        # 空点マスク
        empty = self.get_empty_mask()  # (batch, 19, 19)
        
        # コウ禁止点マスク（簡易版：今回は無視してもOK）
        # Tromp-Taylor では厳密にはスーパーコウだが、Phase III.2 では省略
        
        legal_board = empty  # (batch, 19, 19)
        
        # フラット化してパスを追加
        legal_flat = legal_board.view(B, -1)  # (batch, 361)
        pass_legal = torch.ones(B, 1, dtype=torch.bool, device=self.device)
        
        return torch.cat([legal_flat, pass_legal], dim=1)  # (batch, 362)
    
    # ========================================================
    # Move Execution (Tromp-Taylor: No Capture)
    # ========================================================
    
    def play_batch(self, moves: torch.Tensor) -> torch.Tensor:
        """
        バッチで着手を実行（Tromp-Taylor: 捕獲なし）
        
        Args:
            moves: (batch,) - 着手（0-360=座標, 361=パス）
        
        Returns:
            (batch,) - 成功=True
        """
        B = self.batch_size
        is_pass = moves == PASS_TOKEN
        is_board_move = ~is_pass
        
        # パスの処理
        self.consecutive_passes = torch.where(
            is_pass,
            self.consecutive_passes + 1,
            torch.zeros_like(self.consecutive_passes)
        )
        
        # 盤上の着手（石を置くだけ、捕獲なし）
        if is_board_move.any():
            rows, cols = tokens_to_coords_batch(moves)
            current_color = self.turn
            batch_idx = torch.arange(B, device=self.device)
            
            # 石を置く
            for color in [0, 1]:
                mask = is_board_move & (current_color == color)
                if mask.any():
                    self.boards[batch_idx[mask], color, rows[mask], cols[mask]] = 1.0
        
        # 手番を交代
        self.turn = 1 - self.turn
        
        # 履歴に追加
        self._add_to_history(moves)
        
        # 手数をインクリメント
        self.move_count += 1
        
        # 直前の着手を更新
        self.last_move = moves
        
        return torch.ones(B, dtype=torch.bool, device=self.device)
    
    def _add_to_history(self, moves: torch.Tensor):
        """履歴に着手を追加"""
        B = self.batch_size
        idx = self.move_count.clamp(0, self.max_history - 1)
        batch_idx = torch.arange(B, device=self.device)
        self.history[batch_idx, idx] = moves
    
    # ========================================================
    # Game State Queries
    # ========================================================
    
    def is_game_over(self) -> torch.Tensor:
        """終局判定（両者パス）"""
        return self.consecutive_passes >= 2
    
    def compute_score(self, komi: float = 6.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Tromp-Taylor ルールでスコアを計算
        
        Returns:
            black_score: (batch,)
            white_score: (batch,)
            winner: (batch,) - +1=黒勝ち, -1=白勝ち
        """
        return compute_territory(self.boards, komi)
    
    def count_stones(self) -> torch.Tensor:
        """石の数をカウント"""
        black = self.boards[:, 0].sum(dim=(1, 2))
        white = self.boards[:, 1].sum(dim=(1, 2))
        return torch.stack([black, white], dim=1)
    
    def get_history_tensor(self) -> torch.Tensor:
        """履歴テンソルを取得"""
        return self.history
    
    def get_current_sequence(self, max_len: Optional[int] = None) -> torch.Tensor:
        """現在までの手順をシーケンスとして取得"""
        if max_len is None:
            max_len = self.move_count.max().item()
        max_len = min(max_len, self.max_history)
        return self.history[:, :max_len]
    
    # ========================================================
    # Board Reconstruction
    # ========================================================
    
    def replay_history_to_boards_fast(self, history: torch.Tensor) -> torch.Tensor:
        """
        [Vectorized] 履歴から盤面状態を再構成 (One-Hot + Cumsum)
        
        Algorithm:
        1. 手番ごとの石の色（1 or -1）を計算
        2. move index を One-Hot ベクトルに展開
        3. 時間軸方向に累積和 (cumsum) を取り、盤面の遷移を一括計算
        4. 時間軸を1つシフト（boards[t]はt手目 *直前* の状態）
        
        Assumptions:
        - 入力 history は合法手のみを含む（既に石がある場所には打たない）
        - そのため、加算（Cumsum）は代入（Assignment）と等価になる
        """
        seq_len = history.size(0)
        device = self.device
        
        # 1. 手番と石の色を計算
        turns = torch.arange(seq_len, device=device)
        stones = torch.where(turns % 2 == 0, 1.0, -1.0)
        
        # 2. 有効な手（PASS/PAD以外）のマスク作成
        valid_mask = (history < PASS_TOKEN)
        stones = stones * valid_mask.float()
        
        # 3. One-Hot グリッドの作成 [L, 361]
        safe_moves = history.clone()
        safe_moves[~valid_mask] = 0
        
        grid = torch.zeros(seq_len, BOARD_SIZE * BOARD_SIZE, device=device)
        grid.scatter_(1, safe_moves.unsqueeze(1), stones.unsqueeze(1))
        
        # 4. 累積和 (Cumsum) で盤面推移を一括計算
        states = torch.cumsum(grid, dim=0)
        
        # 5. シフト処理
        boards_flat = torch.zeros_like(states)
        boards_flat[1:] = states[:-1]
        
        # 6. Reshape [L, 19, 19]
        return boards_flat.view(seq_len, BOARD_SIZE, BOARD_SIZE)

    # ========================================================
    # Visualization (Debug)
    # ========================================================
    
    def to_string(self, batch_idx: int = 0) -> str:
        """盤面を文字列で表示"""
        board = self.boards[batch_idx]
        black = board[0].cpu().numpy()
        white = board[1].cpu().numpy()
        
        lines = []
        lines.append("   " + " ".join([chr(ord('A') + i) if i < 8 else chr(ord('A') + i + 1) 
                                        for i in range(BOARD_SIZE)]))
        
        for row in range(BOARD_SIZE):
            line = f"{BOARD_SIZE - row:2d} "
            for col in range(BOARD_SIZE):
                if black[row, col] > 0:
                    line += "● "
                elif white[row, col] > 0:
                    line += "○ "
                else:
                    line += ". "
            line += f"{BOARD_SIZE - row:2d}"
            lines.append(line)
        
        lines.append("   " + " ".join([chr(ord('A') + i) if i < 8 else chr(ord('A') + i + 1) 
                                        for i in range(BOARD_SIZE)]))
        
        return "\n".join(lines)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    print("Testing GPUGoEngine (Tromp-Taylor)...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    engine = GPUGoEngine(batch_size=4, device=device)
    
    # 基本テスト
    print(f"\n=== Initial State ===")
    print(f"Board shape: {engine.boards.shape}")
    
    # 着手テスト
    tengen = coord_to_token(9, 9)
    moves = torch.full((4,), tengen, dtype=torch.long, device=device)
    engine.play_batch(moves)
    print(f"Played at tengen")
    print(engine.to_string(0))
    
    # スコア計算テスト
    black_score, white_score, winner = engine.compute_score()
    print(f"\n=== Score ===")
    print(f"Black: {black_score[0].item()}, White: {white_score[0].item()}")
    print(f"Winner: {winner[0].item()}")
    
    print("\nGPUGoEngine (Tromp-Taylor) test complete!")
