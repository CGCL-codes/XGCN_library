from XGCN.utils import io, csr
from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import print_dict, ensure_dir

import numpy as np
import numba
import torch
from scipy.sparse import csr_matrix
import os.path as osp
from tqdm import tqdm


def calc_beta_from_degree(di, dj):
    beta_i = np.sqrt(di + 1) / (di + 1e-8)
    beta_j = 1 / np.sqrt(dj + 1)
    return beta_i, beta_j


@numba.jit(nopython=True, parallel=True)
def calc_ii_topk(indptr, indices, data, beta_i, beta_j, topk, item_ids):
    n = len(item_ids)
    degree_ii_neighbors = np.zeros((n, topk), dtype=np.int32)
    degree_ii_similarity_scores = np.zeros((n, topk), dtype=np.float32)
    beta_ii_neighbors = np.zeros((n, topk), dtype=np.int32)
    beta_ii_similarity_scores = np.zeros((n, topk), dtype=np.float32)
    for t in numba.prange(n):
        item_id = item_ids[t]
        ptr_start = indptr[item_id]
        ptr_end = indptr[item_id + 1]
        
        neighbors = indices[ptr_start : ptr_end]
        degrees = data[ptr_start : ptr_end]
        sim_score = beta_i[item_id] * beta_j[neighbors] * degrees
        
        sorted_idx = np.argsort(degrees)
        _topk_idx = sorted_idx[-topk:][::-1]
        _topk = len(_topk_idx)
        
        degree_ii_neighbors[t][:_topk] = neighbors[_topk_idx]
        degree_ii_similarity_scores[t][:_topk] = degrees[_topk_idx]
        
        sorted_idx = np.argsort(sim_score)
        _topk_idx = sorted_idx[-topk:][::-1]
        _topk = len(_topk_idx)
        
        beta_ii_neighbors[t][:_topk] = neighbors[_topk_idx]
        beta_ii_similarity_scores[t][:_topk] = sim_score[_topk_idx]
    return degree_ii_neighbors, degree_ii_similarity_scores, beta_ii_neighbors, beta_ii_similarity_scores


@numba.jit(nopython=True, parallel=True)
def calc_sum_of_scores(num_items, ii_neighbors, ii_similarity_scores):
    S = np.zeros(num_items, dtype=np.float32)
    indices = ii_neighbors.reshape(-1)
    scores = ii_similarity_scores.reshape(-1)
    for i in numba.prange(len(indices)):
        idx = indices[i]
        s = scores[i]
        S[idx] += s
    return S


def main():
    
    # input: train_undi_graph, train_graph
    # output: constrain_mat, ii_topk_neighbors, ii_topk_similarity_scores
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    graph_type = info['graph_type']
    num_nodes = info['num_nodes']
    if graph_type == 'user-item':
        num_users = info['num_users']
        num_items = info['num_items']
    
    # calc constraint_mat
    print("## calc beta_users, beta_items ...")
    indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
    undi_indptr, undi_indices = csr.get_undirected(indptr, indices)
    
    if graph_type == 'user-item':
        users_degree = (undi_indptr[1:num_users+1] - undi_indptr[0:num_users]).reshape(-1)
        items_degree = (undi_indptr[num_users+1:] - undi_indptr[num_users:-1]).reshape(-1)
    else:
        users_degree = (undi_indptr[1:] - undi_indptr[:-1]).reshape(-1)
        items_degree = users_degree
    
    beta_users, beta_items = calc_beta_from_degree(users_degree, items_degree)
    constraint_mat = {
        'beta_users': beta_users,
        'beta_items': beta_items
    }
    io.save_pickle(osp.join(results_root, 'constrain_mat.pkl'), constraint_mat)
    
    # calc topk items
    print("## calc A^A ...")
    if graph_type == 'user-item':
        del undi_indptr
        A = csr_matrix((np.ones(len(indices), dtype=np.float32),
                        indices - num_users, 
                        indptr[:num_users+1]),
                       shape=(num_users, num_items))
        A2 = A.T.dot(A)
        del indptr, indices, A
    else:
        A = csr_matrix((np.ones(len(undi_indices), dtype=np.float32),
                        undi_indices, 
                        undi_indptr),
                       shape=(num_nodes, num_nodes))
        A2 = A.T.dot(A)
        del undi_indptr, undi_indices, A
    
    di = np.array(A2.sum(axis=-1)).reshape(-1)
    dj = di  # A2 is symmetric
    beta_i, beta_j = calc_beta_from_degree(di, dj)
    
    topk = config['topk']
    if graph_type == 'user-item':
        _num_items = num_items
    else:
        _num_items = num_nodes
    
    degree_ii_neighbors = np.zeros((_num_items, topk), dtype=np.int32)
    degree_ii_similarity_scores = np.zeros((_num_items, topk), dtype=np.float32)    
    beta_ii_neighbors = np.zeros((_num_items, topk), dtype=np.int32)
    beta_ii_similarity_scores = np.zeros((_num_items, topk), dtype=np.float32)
    
    print("## calc topk items ...")
    dl = torch.utils.data.DataLoader(torch.arange(_num_items), batch_size=512)
    for item_ids in tqdm(dl):
        item_ids = item_ids.numpy()
        batch_data = calc_ii_topk(A2.indptr, A2.indices, A2.data, 
                                  beta_i, beta_j, topk, item_ids)
        degree_ii_neighbors[item_ids] = batch_data[0]
        degree_ii_similarity_scores[item_ids] = batch_data[1]
        beta_ii_neighbors[item_ids] = batch_data[2]
        beta_ii_similarity_scores[item_ids] = batch_data[3]
    
    save_root = osp.join(results_root, 'ii_degree_topk')
    ensure_dir(save_root)
    io.save_pickle(osp.join(save_root, 'ii_topk_neighbors.np.pkl'), degree_ii_neighbors)
    io.save_pickle(osp.join(save_root, 'ii_topk_similarity_scores.np.pkl'), degree_ii_similarity_scores)
    
    save_root = osp.join(results_root, 'beta_score_topk')
    ensure_dir(save_root)
    io.save_pickle(osp.join(save_root, 'ii_topk_neighbors.np.pkl'), beta_ii_neighbors)
    io.save_pickle(osp.join(save_root, 'ii_topk_similarity_scores.np.pkl'), beta_ii_similarity_scores)


if __name__ == '__main__':

    main()
