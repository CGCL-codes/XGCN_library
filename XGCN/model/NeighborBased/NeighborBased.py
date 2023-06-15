from XGCN.data import io, csr
from XGCN.model.module.mask_neighbor_score import mask_neighbor_score

import os.path as osp
import numpy as np
import numba


@numba.jit(nopython=True)
def get_all_1_hop_and_2_hop_nodes(indptr, indices, u):
    nodes = set()
    one_hop_nodes = csr.get_neighbors(indptr, indices, u)
    nodes.update(list(one_hop_nodes))
    for v in one_hop_nodes:
        nodes.update(csr.get_neighbors(indptr, indices, v))
    return np.array(list(nodes))


@numba.jit(nopython=True)
def calc_CN_score(indptr, indices, u, v, use_jaccard=False):
    nu = csr.get_neighbors(indptr, indices, u)
    nv = csr.get_neighbors(indptr, indices, v)
    nu = set(list(nu))
    nv = set(list(nv))
    score = len(nu & nv)  # common neighbor
    if use_jaccard:
        score /= len(nu | nv)  # Jaccard score
    score += np.random.randn() * 0.0001
    return score


@numba.jit(nopython=True)
def CN_for_a_src_node_on_whole_graph(indptr, indices, src, use_jaccard=False):
    nodes12 = get_all_1_hop_and_2_hop_nodes(indptr, indices, src)
    
    scores_of_nodes12 = np.empty(shape=(len(nodes12),), dtype=np.float32)
    for i, v in enumerate(nodes12):
        scores_of_nodes12[i] = calc_CN_score(indptr, indices, src, v, use_jaccard)
   
    num_nodes = len(indptr) - 1
    all_scores = -1 + np.random.randn(num_nodes) * 0.0001
    all_scores[nodes12] = scores_of_nodes12

    return all_scores


@numba.jit(nopython=True, parallel=True)
def CN_for_a_batch_src_node_on_whole_graph(indptr, indices, batch_src, use_jaccard=False):
    num_nodes = len(indptr) - 1
    S = np.empty(shape=(len(batch_src), num_nodes), dtype=np.float32)
    for i in numba.prange(len(batch_src)):
        src = batch_src[i]
        S[i] = CN_for_a_src_node_on_whole_graph(indptr, indices, src, use_jaccard)
    return S


@numba.jit(nopython=True)
def CN_for_a_group_of_src_pos_neg(indptr, indices, src, pos, neg):
    S = np.empty(shape=(1 + len(neg),), dtype=np.float32)
    for i, v in enumerate(np.concatenate((np.array([pos,]), neg))):
        S[i] = calc_CN_score(indptr, indices, src, v)
    return S


@numba.jit(nopython=True, parallel=True)
def CN_for_a_batch_of_src_pos_neg(indptr, indices, batch_src, batch_pos, batch_neg):
    S = np.empty(shape=(len(batch_src), 1 + batch_neg.shape[-1]), dtype=np.float32)
    for i in numba.prange(len(batch_src)):
        S[i] = CN_for_a_group_of_src_pos_neg(
            indptr, indices, batch_src[i], batch_pos[i], batch_neg[i])
    return S


class NeighborBased:
    
    def __init__(self, config):
        self.config = config
        data_root = self.config['data_root']
        self._indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
        self._indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))

        # use undirected graph
        print("# use undirected graph")
        self.indptr, self.indices = csr.get_undirected(self._indptr, self._indices)
        # self.indptr, self.indices = self._indptr, self._indices

    def _eval_a_batch(self, batch_data, eval_type):
        return {
            'whole_graph_multi_pos': self._eval_whole_graph_multi_pos,
            'whole_graph_one_pos': self._eval_whole_graph_one_pos,
            'one_pos_k_neg': self._eval_one_pos_k_neg
        }[eval_type](batch_data)
    
    def infer_all_target_score(self, src, mask_nei=True):
        all_target_score = CN_for_a_batch_src_node_on_whole_graph(
            self.indptr, self.indices, src, self.config['use_jaccard']
        )     
        if mask_nei:
            self.mask_neighbor_score(src, all_target_score)
        return all_target_score  
     
    def mask_neighbor_score(self, src, all_target_score):
        mask_neighbor_score(self._indptr, self._indices,
            src, all_target_score
        )
        
    def _eval_whole_graph_multi_pos(self, batch_data):
        src, _ = batch_data
        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        return all_target_score
    
    def _eval_whole_graph_one_pos(self, batch_data):
        src, pos = batch_data

        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        
        pos_score = np.empty((len(src),), dtype=np.float32)
        for i in range(len(src)):
            pos_score[i] = all_target_score[i][pos[i]]
        pos_neg_score = np.concatenate((pos_score.reshape(-1, 1), all_target_score), axis=-1)
        
        return pos_neg_score

    def _eval_one_pos_k_neg(self, batch_data):
        src, pos, neg = batch_data
        src, pos, neg = src.numpy(), pos.numpy(), neg.numpy()
        pos_neg_score = CN_for_a_batch_of_src_pos_neg(
            self.indptr, self.indices, batch_src=src, batch_pos=pos, batch_neg=neg
        )
        return pos_neg_score
