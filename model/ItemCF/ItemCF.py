from utils import io
from utils.metric import all_metrics, multi_pos_all_metrics

import numpy as np
import numba
from scipy.sparse import csr_matrix
import os.path as osp


@numba.jit(nopython=True, parallel=True)
def _batch_item_cf(src, pos, num_users, ui_indptr, ui_indices, ii_indptr, ii_indices, ii_weights, i_weights_sum):
    num_items = len(i_weights_sum)
    batch_size = len(src)
    pos_neg_score = np.empty((batch_size, 1 + num_items), dtype=np.float32)
    for i in numba.prange(batch_size):
        u = src[i]
        # u_pos = pos[i] - num_users
        u_pos = pos[i]
        u_items = ui_indices[ui_indptr[u] : ui_indptr[u + 1]] - num_users
        all_item_scores = _item_cf(u_items, ii_indptr, ii_indices, ii_weights, i_weights_sum)
        all_item_scores[u_items] = -999999  # mask existing interactions
        pos_neg_score[i][0] = all_item_scores[u_pos]
        pos_neg_score[i][1:] = all_item_scores
    return pos_neg_score


@numba.jit(nopython=True, parallel=True)
def _batch_item_cf_return_all_item_scores(src, num_users, ui_indptr, ui_indices, ii_indptr, ii_indices, ii_weights, i_weights_sum):
    num_items = len(i_weights_sum)
    batch_size = len(src)
    all_item_scores = np.empty((batch_size, num_items), dtype=np.float32)
    for i in numba.prange(batch_size):
        u = src[i]
        u_items = ui_indices[ui_indptr[u] : ui_indptr[u + 1]] - num_users
        u_all_item_scores = _item_cf(u_items, ii_indptr, ii_indices, ii_weights, i_weights_sum)
        u_all_item_scores[u_items] = -999999  # mask existing interactions
        all_item_scores[i] = u_all_item_scores
    return all_item_scores


@numba.jit(nopython=True)
def _item_cf(items, ii_indptr, ii_indices, ii_weights, i_weights_sum):
    all_item_scores = np.zeros(len(i_weights_sum), dtype=np.float32)
    for item_id in items:
        ptr_start = ii_indptr[item_id]
        ptr_end = ii_indptr[item_id + 1]
        
        nei = ii_indices[ptr_start : ptr_end]
        wei = ii_weights[ptr_start : ptr_end]
        
        all_item_scores[nei] += wei
        
    # all_item_scores /= i_weights_sum  # normalization
    all_item_scores += np.random.randn(len(i_weights_sum)) * 1e-8
    return all_item_scores


class ItemCF:
    
    def __init__(self, config, data=None):
        self.config = config
        self.data = data
        
        self.ui_indptr = io.load_pickle(osp.join(self.config['data_root'], 'train_csr_indptr.pkl'))
        self.ui_indices = io.load_pickle(osp.join(self.config['data_root'], 'train_csr_indices.pkl'))
        
        info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
        num_users = info['num_users']
        num_items = info['num_items']
        self.num_users = num_users
        
        print("## calc A^A...")
        A = csr_matrix((np.ones(len(self.ui_indices), dtype=np.float32),
                        self.ui_indices - num_users, 
                        self.ui_indptr[:num_users+1]),
                       shape=(num_users, num_items))
        A2 = A.T.dot(A)
        
        # di = np.array(A2.sum(axis=-1)).reshape(-1)
        # dj = di  # A2 is symmetric
        
        self.ii_indptr = A2.indptr
        self.ii_indices = A2.indices
        self.ii_weights = A2.data  # A2(i, j) is #co-occurence of item_i and item_j
        self.i_weights_sum = np.array(A2.sum(axis=-1)).reshape(-1)
        
        self.one_pos = None  # set by the eval_dl
        self.mask_nei = True
        
    def eval_a_batch(self, batch_data):
        batch_results = self.inference(batch_data)
        return batch_results
    
    def inference(self, batch_data):
        eval_on_whole_graph = len(batch_data) == 2
        
        if not eval_on_whole_graph:
            assert 0
        else:  # eval_on_whole_graph
            src, pos = batch_data
            num_batch_samples = len(src)
            
            if self.one_pos:
                pos_neg_score = _batch_item_cf(
                    src, pos, self.num_users,
                    self.ui_indptr, self.ui_indices, 
                    self.ii_indptr, self.ii_indices, self.ii_weights, 
                    self.i_weights_sum
                )
                batch_results = all_metrics(pos_neg_score)
            else:
                all_item_scores = _batch_item_cf_return_all_item_scores(
                    src, self.num_users, self.ui_indptr, self.ui_indices, 
                    self.ii_indptr, self.ii_indices, self.ii_weights, self.i_weights_sum
                )
                batch_results = multi_pos_all_metrics(pos, all_item_scores)
                
        return batch_results, num_batch_samples
