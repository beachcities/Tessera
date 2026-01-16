
import torch
import torch.nn.functional as F
from typing import Tuple

BOARD_SIZE = 19

def compute_chain_ids(board, color, max_iterations=10):
    B = board.shape[0]
    device = board.device
    color_mask = board[:, color, :, :]
    row_idx = torch.arange(BOARD_SIZE, device=device).view(1, -1, 1).expand(B, BOARD_SIZE, BOARD_SIZE)
    col_idx = torch.arange(BOARD_SIZE, device=device).view(1, 1, -1).expand(B, BOARD_SIZE, BOARD_SIZE)
    batch_offset = torch.arange(B, device=device).view(-1, 1, 1) * 400
    ids = (row_idx * BOARD_SIZE + col_idx + 1 + batch_offset) * color_mask
    for _ in range(max_iterations):
        ids_padded = ids.view(B, 1, BOARD_SIZE, BOARD_SIZE)
        neighbor_max = F.max_pool2d(F.pad(ids_padded, (1,1,1,1), mode="constant", value=0), kernel_size=3, stride=1, padding=0).squeeze(1)
        new_ids = torch.where(color_mask > 0, torch.maximum(ids, neighbor_max), ids)
        if torch.equal(new_ids, ids):
            break
        ids = new_ids
    return ids

def compute_liberty_map(board, chain_ids):
    B = board.shape[0]
    device = board.device
    occupied = board[:, 0] + board[:, 1]
    empty_mask = (occupied == 0).float()
    kernel = torch.tensor([[0,1,0],[1,0,1],[0,1,0]], dtype=torch.float32, device=device).view(1,1,3,3)
    liberty_map = torch.zeros(B, BOARD_SIZE, BOARD_SIZE, device=device)
    unique_ids = chain_ids.unique()
    unique_ids = unique_ids[unique_ids > 0]
    for uid in unique_ids:
        chain_mask = (chain_ids == uid).float()
        chain_padded = chain_mask.view(B, 1, BOARD_SIZE, BOARD_SIZE)
        adjacent = F.conv2d(F.pad(chain_padded, (1,1,1,1), mode="constant", value=0), kernel, padding=0).squeeze(1)
        liberties = ((adjacent > 0) & (empty_mask > 0)).sum()
        liberty_map = torch.where(chain_ids == uid, liberties.float(), liberty_map)
    return liberty_map

def remove_captured_stones(board, color):
    B = board.shape[0]
    chain_ids = compute_chain_ids(board, color)
    liberty_map = compute_liberty_map(board, chain_ids)
    captured_mask = (chain_ids > 0) & (liberty_map == 0)
    captured_counts = captured_mask.sum(dim=(1, 2))
    new_board = board.clone()
    new_board[:, color] = board[:, color] * (~captured_mask).float()
    return new_board, captured_counts
