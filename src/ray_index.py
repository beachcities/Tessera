"""
Tessera Phase III POC-1a: Ray Index Table Generator
====================================================
8方向 ray-cast のためのインデックステーブルを生成する。

D8対称性を考慮し、直線4方向と斜め4方向でパラメータを共有する設計。

Usage:
    python src/ray_index.py  # テスト実行
"""

import torch
import torch.nn as nn
from typing import Tuple, List

# 盤面サイズ
BOARD_SIZE = 19

# 8方向の定義（D8対称性を考慮した順序）
DIRECTIONS = [
    ( 0,  1),  # 0: 上
    ( 0, -1),  # 1: 下
    ( 1,  0),  # 2: 右
    (-1,  0),  # 3: 左
    ( 1,  1),  # 4: 右上
    ( 1, -1),  # 5: 右下
    (-1,  1),  # 6: 左上
    (-1, -1),  # 7: 左下
]

# D8対称性グループ
LINE_GROUP = [0, 1, 2, 3]  # 直線（上下左右）
DIAG_GROUP = [4, 5, 6, 7]  # 斜め

# 最大距離（盤端から盤端）
MAX_DISTANCE = BOARD_SIZE - 1  # 18


def create_ray_index_table(max_dist: int = MAX_DISTANCE) -> torch.Tensor:
    """
    8方向 ray-cast のインデックステーブルを生成する。
    
    Returns:
        index_table: [8, max_dist, 19, 19, 2]
            - 8: 方向数
            - max_dist: 最大距離
            - 19, 19: 盤面座標
            - 2: (row, col) 座標
            
        範囲外の座標は (-1, -1) で埋める（sentinel）
    """
    index_table = torch.full(
        (8, max_dist, BOARD_SIZE, BOARD_SIZE, 2),
        fill_value=-1,
        dtype=torch.long
    )
    
    for d_idx, (dr, dc) in enumerate(DIRECTIONS):
        for r in range(1, max_dist + 1):  # 距離 1 から max_dist まで
            for i in range(BOARD_SIZE):
                for j in range(BOARD_SIZE):
                    # r 手先の座標
                    ni = i + dr * r
                    nj = j + dc * r
                    
                    # 盤内チェック
                    if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                        index_table[d_idx, r - 1, i, j, 0] = ni
                        index_table[d_idx, r - 1, i, j, 1] = nj
    
    return index_table


def create_flat_index_table(max_dist: int = MAX_DISTANCE) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    gather 用のフラットインデックスとマスクを生成する。
    
    Returns:
        flat_index: [8, max_dist, 19, 19] - フラットインデックス (0-360)
        valid_mask: [8, max_dist, 19, 19] - 有効な座標のマスク (bool)
    """
    index_table = create_ray_index_table(max_dist)
    
    # フラットインデックスに変換 (row * 19 + col)
    flat_index = index_table[..., 0] * BOARD_SIZE + index_table[..., 1]
    
    # 範囲外は 0 に設定（後でマスクする）
    valid_mask = (index_table[..., 0] >= 0)
    flat_index = torch.where(valid_mask, flat_index, torch.zeros_like(flat_index))
    
    return flat_index, valid_mask


class RayIndexBuffer(nn.Module):
    """
    Ray-cast 用のインデックスバッファを保持するモジュール。
    
    学習パラメータではなく、バッファとして保持する。
    """
    
    def __init__(self, max_dist: int = MAX_DISTANCE):
        super().__init__()
        self.max_dist = max_dist
        
        flat_index, valid_mask = create_flat_index_table(max_dist)
        
        # バッファとして登録（学習パラメータではない）
        self.register_buffer('flat_index', flat_index)
        self.register_buffer('valid_mask', valid_mask)
        
    def get_indices(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """インデックスとマスクを取得"""
        return self.flat_index, self.valid_mask
    
    def __repr__(self):
        return (
            f"RayIndexBuffer(\n"
            f"  max_dist={self.max_dist},\n"
            f"  flat_index={self.flat_index.shape},\n"
            f"  valid_mask={self.valid_mask.shape},\n"
            f"  valid_ratio={self.valid_mask.float().mean():.2%}\n"
            f")"
        )


def verify_d8_symmetry(index_buffer: RayIndexBuffer) -> bool:
    """
    D8対称性が正しく生成されているか検証する。
    
    - 90度回転で方向インデックスが正しく対応するか
    - 反転で方向インデックスが正しく対応するか
    """
    flat_index, valid_mask = index_buffer.get_indices()
    
    # テスト: 中央の点 (9, 9) からの各方向を確認
    center = (9, 9)
    
    print(f"\n=== D8 対称性検証 ===")
    print(f"中心点: {center}")
    print()
    
    for d_idx, (dr, dc) in enumerate(DIRECTIONS):
        # 距離 3 での到達点を確認
        dist = 3
        idx = flat_index[d_idx, dist - 1, center[0], center[1]].item()
        valid = valid_mask[d_idx, dist - 1, center[0], center[1]].item()
        
        if valid:
            target_r = idx // BOARD_SIZE
            target_c = idx % BOARD_SIZE
            expected_r = center[0] + dr * dist
            expected_c = center[1] + dc * dist
            
            match = (target_r == expected_r and target_c == expected_c)
            status = "✅" if match else "❌"
            print(f"方向 {d_idx} ({dr:+d}, {dc:+d}): "
                  f"距離 {dist} → ({target_r}, {target_c}) "
                  f"期待値 ({expected_r}, {expected_c}) {status}")
        else:
            print(f"方向 {d_idx} ({dr:+d}, {dc:+d}): 範囲外")
    
    return True


def compute_memory_usage(index_buffer: RayIndexBuffer) -> dict:
    """メモリ使用量を計算"""
    flat_index, valid_mask = index_buffer.get_indices()
    
    # バイト数
    index_bytes = flat_index.numel() * flat_index.element_size()
    mask_bytes = valid_mask.numel() * valid_mask.element_size()
    total_bytes = index_bytes + mask_bytes
    
    return {
        'flat_index_shape': tuple(flat_index.shape),
        'valid_mask_shape': tuple(valid_mask.shape),
        'flat_index_bytes': index_bytes,
        'mask_bytes': mask_bytes,
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 * 1024),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Tessera Phase III POC-1a: Ray Index Table Generator")
    print("=" * 60)
    
    # インデックスバッファ生成
    print("\n[1] インデックスバッファ生成...")
    index_buffer = RayIndexBuffer(max_dist=MAX_DISTANCE)
    print(index_buffer)
    
    # メモリ使用量
    print("\n[2] メモリ使用量...")
    mem = compute_memory_usage(index_buffer)
    print(f"  flat_index: {mem['flat_index_shape']} = {mem['flat_index_bytes']:,} bytes")
    print(f"  valid_mask: {mem['valid_mask_shape']} = {mem['mask_bytes']:,} bytes")
    print(f"  合計: {mem['total_mb']:.2f} MB")
    
    # D8対称性検証
    print("\n[3] D8対称性検証...")
    verify_d8_symmetry(index_buffer)
    
    # 有効率の方向別集計
    print("\n[4] 方向別の有効率...")
    flat_index, valid_mask = index_buffer.get_indices()
    for d_idx, (dr, dc) in enumerate(DIRECTIONS):
        ratio = valid_mask[d_idx].float().mean().item()
        group = "直線" if d_idx in LINE_GROUP else "斜め"
        print(f"  方向 {d_idx} ({dr:+d}, {dc:+d}) [{group}]: {ratio:.1%}")
    
    # GPU 転送テスト
    print("\n[5] GPU 転送テスト...")
    if torch.cuda.is_available():
        index_buffer = index_buffer.cuda()
        flat_index, valid_mask = index_buffer.get_indices()
        print(f"  ✅ GPU 転送成功: {flat_index.device}")
    else:
        print("  ⚠️ CUDA 利用不可")
    
    print("\n" + "=" * 60)
    print("POC-1a 完了: ray_index.py")
    print("=" * 60)
