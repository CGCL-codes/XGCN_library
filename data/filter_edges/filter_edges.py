import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from utils import io
from data.csr_graph_helper import *

import numpy as np
from scipy.sparse import csr_matrix
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']  # graph to remove edges
    results_root = config['results_root']  # new instance
    ensure_dir(results_root)
    
    info = {}
    info['dataset_name'] = config['dataset_name']
    info['dataset_type'] = 'social'
    
    indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
    edge_weights = io.load_pickle(osp.join(data_root, 'edge_weights.pkl'))
    num_nodes = len(indptr) - 1
    info['num_nodes'] = num_nodes
    
    # min_weight = input("please input min edge_weight")
    min_weight = None
    import pdb; pdb.set_trace()  # set min_weight manually
    
    mask = (edge_weights < min_weight)
    
    print("- remove_edges_in_csr ...")
    indptr, indices = remove_edges_in_csr(indptr, indices, mask)

    print("- remove_repeated_edges_in_csr ...")
    indptr, indices = remove_repeated_edges_in_csr(indptr, indices)
    
    print("- get_src_indices ...")
    src_indices = get_src_indices(indptr)
    
    info['num_edges'] = len(indices)
    print("- graph info:")
    print_dict(info)
    io.save_yaml(osp.join(results_root, 'info.yaml'), info)
    
    print("## save directed csr graph ...")
    io.save_pickle(osp.join(results_root, 'train_csr_indptr.pkl'), indptr)
    io.save_pickle(osp.join(results_root, 'train_csr_indices.pkl'), indices)
    io.save_pickle(osp.join(results_root, 'train_csr_src_indices.pkl'), src_indices)

    print("## construct undirected csr graph ...")
    undi_E_src = np.concatenate([src_indices, indices])
    undi_E_dst = np.concatenate([indices, src_indices])
    
    del indptr, indices, src_indices
    
    print("- from_edges_to_csr ...")
    undi_indptr, undi_indices = from_edges_to_csr(undi_E_src, undi_E_dst, num_nodes)
    
    del undi_E_src, undi_E_dst
    
    print("- remove_repeated_edges_in_csr ...")
    undi_indptr, undi_indices = remove_repeated_edges_in_csr(undi_indptr, undi_indices)
    
    print("- get_src_indices ...")
    undi_src_indices = get_src_indices(undi_indptr)

    print("## save undirected csr graph ...")
    io.save_pickle(osp.join(results_root, 'train_undi_csr_indptr.pkl'), undi_indptr)
    io.save_pickle(osp.join(results_root, 'train_undi_csr_indices.pkl'), undi_indices)
    io.save_pickle(osp.join(results_root, 'train_undi_csr_src_indices.pkl'), undi_src_indices)

    print("## done!")


if __name__ == '__main__':
    
    setproctitle.setproctitle('filter_edges-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
