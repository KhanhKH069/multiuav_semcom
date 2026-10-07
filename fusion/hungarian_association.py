"""
hungarian_association.py
------------------------
Module thực hiện liên kết mục tiêu (Target Association) theo đúng Chương 4.1
của báo cáo đề tài.

Quy trình:
    1. Xây dựng Cost Matrix C_{i,j} từ Cosine Similarity giữa:
           E_local  (embedding mới đến từ các UAV)
       và  E_global (embedding toàn cục của các track đang lưu vết)
       Công thức (5) trong báo cáo:
           C_{i,j} = 1 - (E_local_i · E_global_j) / (||E_local_i|| ||E_global_j||)

    2. Áp dụng Hungarian Algorithm (scipy.optimize.linear_sum_assignment) để
       tìm phép ghép cặp song ánh một-một tối thiểu hóa tổng chi phí O(n³).

    3. Lọc các cặp có chi phí vượt ngưỡng → loại bỏ False Positive.

API chính:
    associator = HungarianAssociator(threshold=0.4)
    matched, unmatched_local, unmatched_global = associator.associate(
        local_embeddings,   # np.ndarray [M, 128]
        global_embeddings,  # np.ndarray [N, 128]
    )

Kết quả trả về:
    matched         : List[Tuple[int, int]] — các cặp (local_idx, global_idx)
    unmatched_local : List[int]             — local không khớp → tạo track mới
    unmatched_global: List[int]             — global không được quan sát → có thể bị mất
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except ImportError:  
    _SCIPY_AVAILABLE = False

def cosine_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    similarity = A_norm @ B_norm.T   
    return 1.0 - similarity          

def _greedy_assignment(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    M, N = cost_matrix.shape
    row_ind, col_ind = [], []
    used_cols = set()

    flat_indices = np.argsort(cost_matrix.ravel())
    used_rows = set()
    for idx in flat_indices:
        r, c = divmod(int(idx), N)
        if r not in used_rows and c not in used_cols:
            row_ind.append(r)
            col_ind.append(c)
            used_rows.add(r)
            used_cols.add(c)
        if len(row_ind) == min(M, N):
            break

    return np.array(row_ind, dtype=int), np.array(col_ind, dtype=int)

@dataclass
class AssociationResult:
    
    matched: List[Tuple[int, int]]   
    unmatched_local: List[int]       
    unmatched_global: List[int]      
    cost_matrix: np.ndarray          

class HungarianAssociator:
    
    def __init__(self, threshold: float = 0.4):
        if not 0.0 < threshold <= 2.0:
            raise ValueError(f"threshold phải trong (0, 2], nhận được: {threshold}")
        self.threshold = threshold

    def associate(
        self,
        local_embeddings: np.ndarray,
        global_embeddings: np.ndarray,
    ) -> AssociationResult:
        
        M = len(local_embeddings)
        N = len(global_embeddings)

        if M == 0 and N == 0:
            return AssociationResult([], [], [], np.zeros((0, 0)))
        if M == 0:
            return AssociationResult([], [], list(range(N)), np.zeros((0, N)))
        if N == 0:
            return AssociationResult([], list(range(M)), [], np.zeros((M, 0)))

        cost_matrix = cosine_distance_matrix(
            np.array(local_embeddings, dtype=np.float64),
            np.array(global_embeddings, dtype=np.float64),
        )

        if _SCIPY_AVAILABLE:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = _greedy_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        matched_rows: set[int] = set()
        matched_cols: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= self.threshold:
                matched.append((int(r), int(c)))
                matched_rows.add(int(r))
                matched_cols.add(int(c))

        unmatched_local = [i for i in range(M) if i not in matched_rows]
        unmatched_global = [j for j in range(N) if j not in matched_cols]

        return AssociationResult(
            matched=matched,
            unmatched_local=unmatched_local,
            unmatched_global=unmatched_global,
            cost_matrix=cost_matrix,
        )

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("=" * 55)
    print("Demo HungarianAssociator — Multi-Target Association")
    print("=" * 55)

    rng = np.random.default_rng(42)

    DIM = 128
    num_local = 3
    num_global = 4

    global_emb = rng.standard_normal((num_global, DIM)).astype(np.float64)
    global_emb /= np.linalg.norm(global_emb, axis=1, keepdims=True)

    local_emb = np.vstack([
        global_emb[0] + rng.normal(0, 0.05, DIM),   
        global_emb[2] + rng.normal(0, 0.05, DIM),   
        rng.standard_normal(DIM),                    
    ]).astype(np.float64)
    local_emb /= np.linalg.norm(local_emb, axis=1, keepdims=True)

    assoc = HungarianAssociator(threshold=0.4)
    result = assoc.associate(local_emb, global_emb)

    print(f"\nCost Matrix (Cosine Distance) [M={num_local} x N={num_global}]:")
    print(np.round(result.cost_matrix, 3))
    print(f"\nKết quả ghép cặp: {result.matched}")
    print(f"Local không khớp (→ tạo track mới): {result.unmatched_local}")
    print(f"Global không quan sát được (→ có thể mất): {result.unmatched_global}")

    matched_globals = [c for _, c in result.matched]
    assert 0 in matched_globals, "local[0] phải khớp với global[0]"
    assert 2 in matched_globals, "local[1] phải khớp với global[2]"
    assert 2 in result.unmatched_local, "local[2] (mới) phải không khớp"
    print("\n✅ Tất cả assertions passed — HungarianAssociator hoạt động đúng.")
    print(f"{'scipy' if _SCIPY_AVAILABLE else 'Greedy fallback'} được sử dụng.")

