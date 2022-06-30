import numpy as np
import numba
import random


@numba.jit(nopython=True)
def _get_ppr_scores_dict(indptr, indices, source, num_walks, walk_length, alpha):
    res = numba.typed.Dict.empty(
        key_type=numba.types.int64,
        value_type=numba.types.int32,
    )
    res[source] = 1
    for _ in range(num_walks):
        u = source
        for _ in range(walk_length):
            start, end = indptr[u], indptr[u + 1]
            degree = end - start
            p = random.random()  # p is in [0, 1)
            if p < alpha or degree == 0:
                u = source
                continue
            v = indices[np.random.randint(low=start, high=end)]
            if v in res:
                res[v] += 1
            else:
                res[v] = 1
            u = v
    return res


@numba.jit(nopython=True)
def ppr_for_one_node_on_whole_graph(indptr, indices, source, num_walks, walk_length, alpha):
    num_nodes = len(indptr) - 1
    all_target_score = np.zeros(num_nodes, dtype=np.int32)
    res = _get_ppr_scores_dict(indptr, indices, source, num_walks, walk_length, alpha)
    for nid in res:
        all_target_score[nid] = res[nid]
    return all_target_score


@numba.jit(nopython=True)
def ppr_for_batch_nodes_on_whole_graph(indptr, indices, nids, num_walks, walk_length, alpha):
    num_nodes = len(indptr) - 1
    batch_size = len(nids)
    all_target_score = np.empty((batch_size, num_nodes), dtype=np.int32)
    for i in numba.prange(batch_size):
        source = nids[i]
        all_target_score[i] = ppr_for_one_node_on_whole_graph(
            indptr, indices, source, num_walks, walk_length, alpha
        )
    return all_target_score


@numba.jit(nopython=True)
def ppr_for_one_node(indptr, indices, source, topk, num_walks, walk_length, alpha):
    nei = np.zeros(topk, dtype=np.int64)
    wei = np.zeros(topk, dtype=np.int32)
    
    degree = indptr[source + 1] - indptr[source]
    if degree == 0:
        nei[0] = source
        wei[0] = 1
        return nei, wei
    
    res = _get_ppr_scores_dict(indptr, indices, source, num_walks, walk_length, alpha)
    _nei = np.array(list(res.keys()), dtype=np.int64)
    _wei = np.array(list(res.values()), dtype=np.int32)
    
    ind = np.argsort(_wei)[-topk:][::-1]
    _nei = _nei[ind]
    _wei = _wei[ind]
    
    _topk = len(_nei)
    nei[:_topk] = _nei
    wei[:_topk] = _wei
    
    return nei, wei


@numba.jit(nopython=True, parallel=True)
def ppr_for_batch_nodes(
        indptr, indices, 
        nids, 
        topk, num_walks, walk_length, alpha
    ):
    batch_size = len(nids)
    nei = np.zeros((batch_size, topk), dtype=np.int64)
    wei = np.zeros((batch_size, topk), dtype=np.int32)
    for i in numba.prange(batch_size):
        source = nids[i]
        s_nei, s_wei = ppr_for_one_node(
            indptr, indices, source, topk, num_walks, walk_length, alpha
        )
        nei[i] = s_nei
        wei[i] = s_wei
    return nei, wei
