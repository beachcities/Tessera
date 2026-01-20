"""
Tessera Utilities
=================
共通のヘルパー関数

DEC-009: turn_seq生成のベクトル化版
"""

import torch


def get_turn_sequence(
    history_lengths: torch.Tensor,
    current_turns: torch.Tensor,
    max_seq_len: int,
    device: str
) -> torch.Tensor:
    """
    DEC-009: 履歴から自分(0)か相手(1)かを識別する turn_seq を生成する (Vectorized版)
    
    Args:
        history_lengths: 各バッチの実際の着手手数 [B]
        current_turns: 各バッチの「今から打つ人」(0=黒, 1=白) [B]
        max_seq_len: モデルに渡すシーケンス長 (SEQ_LEN)
        device: 'cuda' or 'cpu'
    
    Returns:
        turn_seq: [B, max_seq_len] (自分=0, 相手=1)
    
    Note:
        - 囲碁は必ず先手が黒なので、idx%2で手番を計算
        - Phase III.2は黒から開始固定
        - 将来のhandicap対応時はcurrent_turnsの計算を呼び出し側で変更
    """
    B = history_lengths.size(0)
    
    # [B, max_seq_len] の位置インデックスを生成
    indices = torch.arange(max_seq_len, device=device).view(1, -1).expand(B, -1)
    
    # 右詰めパディングのオフセット計算
    offset = (max_seq_len - history_lengths.view(-1, 1))
    
    # 物理的な手番（j=0:黒, j=1:白）を計算
    abs_turns = (indices - offset) % 2
    
    # 現在の手番と異なるものを1(Other)とする
    turn_seq = (abs_turns != current_turns.view(-1, 1)).long()
    
    # パディング領域を0(Self)でマスク
    mask = indices < offset
    turn_seq[mask] = 0
    
    return turn_seq
