"""
chain_utils.py - Tromp-Taylor ルール用 GPU テンソル演算

Phase III.2: 捕獲処理を廃止し、終局時の地計算のみに特化
"""

import torch
import torch.nn.functional as F
from typing import Tuple

BOARD_SIZE = 19


def flood_fill_from_stones(stones: torch.Tensor, empty: torch.Tensor, iterations: int = 40) -> torch.Tensor:
    """
    石の位置から空点を通じて到達可能な範囲を計算
    
    Args:
        stones: (batch, 19, 19) - 石の位置
        empty: (batch, 19, 19) - 空点マスク
        iterations: 最大反復回数（19x19盤面では40で十分）
        
    Returns:
        (batch, 19, 19) - 到達可能な空点のマスク
    """
    device = stones.device
    kernel = torch.tensor([[0, 1, 0], 
                           [1, 0, 1], 
                           [0, 1, 0]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

    # 石に隣接する空点から開始
    stones_f = stones.float().unsqueeze(1)  # (batch, 1, 19, 19)
    adjacent_to_stones = F.conv2d(stones_f, kernel, padding=1).squeeze(1) > 0
    
    # 初期到達点：石に隣接する空点
    reached = adjacent_to_stones & empty
    reached_f = reached.float().unsqueeze(1)
    empty_f = empty.float().unsqueeze(1)

    for _ in range(iterations):
        expanded = F.conv2d(reached_f, kernel, padding=1)
        new_reached = ((expanded > 0) | (reached_f > 0)) * empty_f
        
        # 収束判定（全バッチで変化なし）
        if (new_reached == reached_f).all():
            break
        reached_f = new_reached
    
    return reached_f.squeeze(1) > 0


def compute_territory(boards: torch.Tensor, komi: float = 6.5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Tromp-Taylor ルールで地を計算
    
    Args:
        boards: (batch, 2, 19, 19) - [黒, 白] の盤面
        komi: コミ（白に加算）
        
    Returns:
        black_score: (batch,) - 黒のスコア（石 + 地）
        white_score: (batch,) - 白のスコア（石 + 地 + コミ）
        winner: (batch,) - +1=黒勝ち, -1=白勝ち, 0=引き分け
    """
    black = boards[:, 0]  # (batch, 19, 19)
    white = boards[:, 1]  # (batch, 19, 19)
    empty = (black == 0) & (white == 0)

    # 石の数
    black_stones = black.sum(dim=(1, 2))
    white_stones = white.sum(dim=(1, 2))

    # 空点への到達可能性を伝播
    black_reach = flood_fill_from_stones(black, empty)
    white_reach = flood_fill_from_stones(white, empty)

    # 地の判定
    # 黒のみ到達 = 黒地, 白のみ到達 = 白地, 両方到達 = 中立（ダメ）
    black_territory = empty & black_reach & ~white_reach
    white_territory = empty & white_reach & ~black_reach

    # スコア計算
    black_score = black_stones + black_territory.sum(dim=(1, 2)).float()
    white_score = white_stones + white_territory.sum(dim=(1, 2)).float() + komi

    # 勝者判定
    winner = torch.sign(black_score - white_score)
    
    return black_score, white_score, winner


# ============================================================
# 以下は旧コード（捕獲ベース）- 参照用にコメントアウト
# Phase III.2 では使用しない
# ============================================================

# def compute_chain_ids(board, color, max_iterations=10):
#     """連IDを計算（旧コード）"""
#     B = board.shape[0]
#     device = board.device
#     color_mask = board[:, color, :, :]
#     row_idx = torch.arange(BOARD_SIZE, device=device).view(1, -1, 1).expand(B, BOARD_SIZE, BOARD_SIZE)
#     col_idx = torch.arange(BOARD_SIZE, device=device).view(1, 1, -1).expand(B, BOARD_SIZE, BOARD_SIZE)
#     batch_offset = torch.arange(B, device=device).view(-1, 1, 1) * 400
#     ids = (row_idx * BOARD_SIZE + col_idx + 1 + batch_offset) * color_mask
#     for _ in range(max_iterations):
#         ids_padded = ids.view(B, 1, BOARD_SIZE, BOARD_SIZE)
#         neighbor_max = F.max_pool2d(F.pad(ids_padded, (1,1,1,1), mode="constant", value=0), 
#                                     kernel_size=3, stride=1, padding=0).squeeze(1)
#         new_ids = torch.where(color_mask > 0, torch.maximum(ids, neighbor_max), ids)
#         if torch.equal(new_ids, ids):
#             break
#         ids = new_ids
#     return ids

# def compute_liberty_map(board, chain_ids):
#     """呼吸点マップを計算（旧コード）"""
#     B = board.shape[0]
#     device = board.device
#     occupied = board[:, 0] + board[:, 1]
#     empty_mask = (occupied == 0).float()
#     kernel = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32, device=device).view(1,1,3,3)
#     liberty_map = torch.zeros(B, BOARD_SIZE, BOARD_SIZE, device=device)
#     unique_ids = chain_ids.unique()
#     unique_ids = unique_ids[unique_ids > 0]
#     for uid in unique_ids:
#         chain_mask = (chain_ids == uid).float()
#         chain_padded = chain_mask.view(B, 1, BOARD_SIZE, BOARD_SIZE)
#         adjacent = F.conv2d(F.pad(chain_padded, (1,1,1,1), mode="constant", value=0), 
#                            kernel, padding=0).squeeze(1)
#         liberties = ((adjacent > 0) & (empty_mask > 0)).sum()
#         liberty_map = torch.where(chain_ids == uid, liberties.float(), liberty_map)
#     return liberty_map

# def remove_captured_stones(board, color):
#     """捕獲された石を除去（旧コード）"""
#     B = board.shape[0]
#     chain_ids = compute_chain_ids(board, color)
#     liberty_map = compute_liberty_map(board, chain_ids)
#     captured_mask = (chain_ids > 0) & (liberty_map == 0)
#     captured_counts = captured_mask.sum(dim=(1, 2))
#     new_board = board.clone()
#     new_board[:, color] = board[:, color] * (~captured_mask).float()
#     return new_board, captured_counts
