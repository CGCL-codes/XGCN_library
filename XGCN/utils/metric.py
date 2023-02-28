from XGCN.utils.utils import combine_dict_list_and_calc_mean

import numpy as np
import numba
from numba.typed import Dict
from numba.core import types


@numba.jit(nopython=True)
def _get_ndcg_weights(length):
    ndcg_weights = 1 / np.log2(np.arange(2, 2 + 100))
    if length > len(ndcg_weights):
        ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights[:length]


@numba.jit(nopython=True)
def _get_mrr_weights(length):
    mrr_weights = 1 / np.arange(1, 1 + 100)
    if length > len(mrr_weights):
        mrr_weights = 1 / np.arange(2, length + 2)
    return mrr_weights[:length]


@numba.jit(nopython=True)
def get_rank(A):
    rank = np.empty(len(A), dtype=np.int32)
    for i in range(len(A)):
        a = A[i]
        key = a[0]
        r = 0
        for j in range(1, len(a)):
            if a[j] > key:
                r += 1
        rank[i] = r
    return rank


def one_pos_metrics(S):
    num_samples = S.shape[0]
    num_scores = S.shape[1]
    
    # add small noises
    S += np.random.uniform(low=-1e-6, high=1e-6, size=S.shape)
    
    rank = get_rank(S)
    
    # top1 = rank == 0
    # top3 = rank < 3
    top20 = rank < 20
    top50 = rank < 50
    top100 = rank < 100
    top300 = rank < 300
    
    results = {
        "auc": (num_scores - 1 - rank).mean() / (num_scores - 1)
    }
    
    # ndcg
    w = _get_ndcg_weights(num_scores)
    results.update({
        "ndcg": w[rank].sum() / num_samples,
        # "n1": top1.mean(),
        # "n3": w[rank[top3]].sum() / num_samples,
        # "n10": w[rank[top10]].sum() / num_samples,
        "n20": w[rank[top20]].sum() / num_samples,
        "n50": w[rank[top50]].sum() / num_samples,
        "n100": w[rank[top100]].sum() / num_samples,
        "n300": w[rank[top300]].sum() / num_samples,
    })
    
    # recall
    results.update({
        "r20": top20.mean(),
        "r50": top50.mean(),
        "r100": top100.mean(),
        "r300": top300.mean(),
    })
    
    # mrr
    w = _get_mrr_weights(num_scores)
    results.update({
        "mrr": w[rank].sum() / num_samples,
    })

    return results


def whole_graph_multi_pos_metrics(pos: list, all_target_score):
    results_dict_list = _new_multi_pos_all_metrics(pos, all_target_score)
    results = combine_dict_list_and_calc_mean(results_dict_list)
    return results


# @numba.jit(nopython=True)
def argtopk(a, k):
    if k == 1:
        return np.array([np.argmax(a)])
    else:
        ind = np.argpartition(a, -k)[-k:]
        return ind[np.argsort(a[ind])][::-1]


# @numba.jit(nopython=True, parallel=True)
def _new_multi_pos_all_metrics(pos_list, all_target_score):
    results_dict_list = [
        Dict.empty(key_type=types.unicode_type, value_type=types.float32)
        for _ in range(len(pos_list))
    ]
    
    all_target_score += np.random.uniform(low=-1e-6, high=1e-6, size=all_target_score.shape)
    
    topk_list = [20, 50, 100, 300]
    max_k = topk_list[-1]
    
    ndcg_weights = 1 / np.log2(np.arange(2, max_k + 2))
    
    # for i in numba.prange(len(pos_list)):
    for i in range(len(pos_list)):
        pos = pos_list[i]
        pos_set = set(list(pos))
        topk_id = argtopk(all_target_score[i], max_k)
        
        ground_truth_label = np.zeros(max_k)
        ground_truth_label[:len(pos)] = 1
        
        pred_label = np.zeros(max_k)
        for j in range(max_k):
            v = topk_id[j]
            if v in pos_set:
                pred_label[j] = 1
        
        results_dict = {}
        
        # calc recall
        for k in topk_list:
            results_dict['r' + str(k)] = pred_label[:k].sum() / ground_truth_label.sum()
        
        # calc ndcg
        s = pred_label * ndcg_weights
        truth_s = ground_truth_label * ndcg_weights
        for k in topk_list:
            results_dict['n' + str(k)] = s[:k].sum() / truth_s[:k].sum()        

        for key in results_dict:
            results_dict_list[i][key] = results_dict[key]
    
    return results_dict_list
