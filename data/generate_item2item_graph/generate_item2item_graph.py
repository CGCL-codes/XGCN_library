import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from utils import io

import numpy as np
from scipy.sparse import csr_matrix
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    
    ui_indptr = io.load_pickle(osp.join(data_root, 'train_csr_indptr.pkl'))
    ui_indices = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    ui_src_indices = io.load_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'))
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    num_users = info['num_users']
    num_items = info['num_items']
    
    print("## calc A^T * A...")
    if config['use_degree_norm']:
        print("# use_degree_norm: np.sqrt(1 / (src_degree * dst_degree))")
        undi_indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
        all_degrees = undi_indptr[1:] - undi_indptr[:-1]
        d_src = all_degrees[ui_src_indices]
        d_dst = all_degrees[ui_indices]
        edge_weights = np.sqrt(1 / (d_src * d_dst + 1e-8))
    else:
        edge_weights = np.ones(len(ui_indices), dtype=np.float32)
    A = csr_matrix((edge_weights,
                    ui_indices - num_users, 
                    ui_indptr[:num_users+1]),
                    shape=(num_users, num_items))
    A2 = A.T.dot(A)

    print("## save...")
    io.save_pickle(osp.join(results_root, 'indptr.pkl'), A2.indptr)
    io.save_pickle(osp.join(results_root, 'indices.pkl'), A2.indices)
    io.save_pickle(osp.join(results_root, 'edge_weights.pkl'), A2.data)


if __name__ == '__main__':
    
    setproctitle.setproctitle('generate_item2item_graph-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
