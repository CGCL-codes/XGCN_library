from .query import get_neighbors

import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def sample_one_strict_neg(neg_low, neg_high, src: np.ndarray,
                            indptr: np.ndarray,
                            indices: np.ndarray):
    # ensure the negative sample is not the src node's neighbor
    neg = np.zeros(len(src), dtype=src.dtype)
    for i in numba.prange(len(src)):
        u = src[i]
        u_nei = get_neighbors(indptr, indices, u)
        u_neg = np.random.randint(neg_low, neg_high)
        while u_neg in u_nei:
            u_neg = np.random.randint(neg_low, neg_high)
        neg[i] = u_neg
    return neg


@numba.jit(nopython=True, parallel=True)
def sample_multi_strict_neg(neg_low, neg_high, num_neg, src: np.ndarray,
                              indptr: np.ndarray,
                              indices: np.ndarray):
    # ensure the negative sample is not the src node's neighbor
    neg = np.zeros((len(src), num_neg), dtype=src.dtype)
    for i in numba.prange(len(src)):
        u = src[i]
        u_nei = get_neighbors(indptr, indices, u)
        u_nei_set = set(list(u_nei))
        u_neg = np.zeros(num_neg, dtype=np.int64)
        cnt = 0
        while cnt < num_neg:
            _neg = np.random.randint(neg_low, neg_high)
            if _neg not in u_nei_set:
                u_neg[cnt] = _neg
                cnt += 1
        neg[i] = u_neg
    return neg
