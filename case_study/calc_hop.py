import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from data.csr_graph_helper import get_neighbors

import numpy as np
import numba
import torch
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


@numba.jit(nopython=True)
def calc_hop(indptr, indices, u, v_list):
    hops = np.empty(shape=(len(v_list),), dtype=np.int32)
    visited = set(np.array([u], dtype=np.int32))

    # get 1-hop
    nei = get_neighbors(indptr, indices, u)
    hop1 = set(nei) - visited
    visited.update(hop1)

    # get 2-hop
    hop2 = set()
    for node in hop1:
        nei = get_neighbors(indptr, indices, node)
        hop2.update(set(nei))
    hop2 = hop2 - visited
    visited.update(hop2)

    # get 3-hop
    hop3 = set()
    for node in hop2:
        nei = get_neighbors(indptr, indices, node)
        hop3.update(set(nei))
    hop3 = hop3 - visited
    
    for j, v in enumerate(v_list):
        if v == u:
            h = 0
        elif v in hop1:
            h = 1
        elif v in hop2:
            h = 2
        elif v in hop3:
            h = 3
        else:
            h = -1
        hops[j] = h
        
    return hops


@numba.jit(nopython=True, parallel=True)
def calc_hop_for_a_batch(indptr, indices, X):
    H = np.empty(shape=(len(X), X.shape[1] - 1), dtype=np.int32)
    for i in numba.prange(len(X)):
        H[i] = calc_hop(indptr, indices, X[i][0], X[i][1:])
    return H


def main():
    
    config = parse_arguments()
    data_root = config['data_root']
    
    X = io.load_pickle(config['file_input'])
    '''
    X: numpy array, top-k reco
    [[src0, dst1, dst2, ..., dst_k],
     [src1, ..., ],
     ...
    ]
    '''
    
    indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
    
    dl = torch.utils.data.DataLoader(dataset=X, batch_size=128)
    H = np.empty(shape=(len(X), X.shape[1] - 1), dtype=np.int32)
    st = 0
    for batch_X in tqdm(dl):
        batch_X = batch_X.numpy()
        batch_H = calc_hop_for_a_batch(indptr, indices, batch_X)
        H[st : st + len(batch_H)] = batch_H
        st += len(batch_H)
    
    io.save_pickle(config['file_output'], H)
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    
    setproctitle.setproctitle('calc_hop-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
