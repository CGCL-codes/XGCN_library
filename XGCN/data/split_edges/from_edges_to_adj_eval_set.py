from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils import io

import numpy as np
import os.path as osp


def from_edges_to_adj_eval_set(edges):
    adj = {}
    for e in edges:
        u, v = e[0], e[1]
        if u in adj:
            adj[u].append(v)
        else:
            adj[u] = [v]

    src = []
    pos_list = []
    for u in adj:
        src.append(u)
        pos_list.append(np.array(adj[u]))
    src = np.array(src)
    
    return {'src': src, 'pos_list': pos_list}


def main():
    
    config = parse_arguments()
    data_root = config['data_root']
    num_validation = config['num_validation']
    
    pos_edges = io.load_pickle(osp.join(data_root, 'pos_edges.pkl'))
    
    val_edges = pos_edges[:num_validation]
    test_edges = pos_edges[num_validation:]

    val_set = from_edges_to_adj_eval_set(val_edges)
    test_set = from_edges_to_adj_eval_set(test_edges)
    
    io.save_pickle(osp.join(data_root, 'val_set.pkl'), val_set)
    io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)


if __name__ == '__main__':
    
    main()
