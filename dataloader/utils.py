import data.csr_graph_helper as csr_helper

import numpy as np
import numba


@numba.jit(nopython=True)
def _sample_one_strict_neg(neg_low, neg_high, u_nei):
    u_neg = np.random.randint(neg_low, neg_high)
    while u_neg in u_nei:
        u_neg = np.random.randint(neg_low, neg_high)
    return u_neg


@numba.jit(nopython=True)
def _sample_multi_strict_neg(neg_low, neg_high, u_nei, num_neg):
    u_nei_set = set(list(u_nei))
    u_neg = np.zeros(num_neg, dtype=np.int64)
    cnt = 0
    while cnt < num_neg:
        _neg = np.random.randint(neg_low, neg_high)
        if _neg not in u_nei_set:
            u_neg[cnt] = _neg
            cnt += 1
    return u_neg


@numba.jit(nopython=True, parallel=True)
def generate_one_strict_neg(neg_low, neg_high, src: np.ndarray,
                            csr_graph_indptr: np.ndarray,
                            csr_graph_indices: np.ndarray):
    # ensure the negative sample is not the src node's neighbor
    neg = np.zeros(len(src), dtype=src.dtype)
    for i in numba.prange(len(src)):
        u = src[i]
        u_nei = csr_helper.get_neighbors(csr_graph_indptr, csr_graph_indices, u)
        neg[i] = _sample_one_strict_neg(neg_low, neg_high, u_nei)
    return neg


@numba.jit(nopython=True, parallel=True)
def generate_multi_strict_neg(neg_low, neg_high, num_neg, src: np.ndarray,
                              csr_graph_indptr: np.ndarray,
                              csr_graph_indices: np.ndarray):
    # ensure the negative sample is not the src node's neighbor
    neg = np.zeros((len(src), num_neg), dtype=src.dtype)
    for i in numba.prange(len(src)):
        u = src[i]
        u_nei = csr_helper.get_neighbors(csr_graph_indptr, csr_graph_indices, u)
        neg[i] = _sample_multi_strict_neg(neg_low, neg_high, u_nei, num_neg)
    return neg


@numba.jit(nopython=True, parallel=True)
def generate_pos_and_one_strict_neg(
    neg_low, neg_high, src: np.ndarray,
    csr_graph_indptr: np.ndarray,
    csr_graph_indices: np.ndarray):
    
    pos = np.zeros(src.shape, dtype=src.dtype)
    neg = np.zeros(src.shape, dtype=src.dtype)
    for i in numba.prange(len(src)):
        u = src[i]
        u_nei = csr_helper.get_neighbors(csr_graph_indptr, csr_graph_indices, u)
        
        u_neg = np.random.randint(neg_low, neg_high)
        while u_neg in u_nei:
            u_neg = np.random.randint(neg_low, neg_high)
        neg[i] = u_neg
        
        if len(u_nei) > 0:
            pos_idx = np.random.randint(0, len(u_nei), 1)[0]
            pos[i] = u_nei[pos_idx]
        else:
            pos[i] = u
        
    return pos, neg


@numba.jit(nopython=True, parallel=True)
def generate_src_pos_and_one_strict_neg(
    src_low, src_high,
    neg_low, neg_high, batch_size,
    csr_graph_indptr: np.ndarray,
    csr_graph_indices: np.ndarray):
    
    src = np.zeros(batch_size, dtype=np.int64)
    pos = np.zeros(batch_size, dtype=np.int64)
    neg = np.zeros(batch_size, dtype=np.int64)
    for i in numba.prange(batch_size):
        while True:
            u = np.random.randint(src_low, src_high)
            u_nei = csr_helper.get_neighbors(csr_graph_indptr, csr_graph_indices, u)
            if len(u_nei) > 0:
                break
        src[i] = u
        
        pos_idx = np.random.randint(0, len(u_nei), 1)[0]
        pos[i] = u_nei[pos_idx]
        
        u_neg = np.random.randint(neg_low, neg_high)
        while u_neg in u_nei:
            u_neg = np.random.randint(neg_low, neg_high)
        neg[i] = u_neg
        
    return src, pos, neg
