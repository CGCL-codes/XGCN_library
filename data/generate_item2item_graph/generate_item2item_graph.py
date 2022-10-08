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
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    num_users = info['num_users']
    num_items = info['num_items']
    
    print("## calc A^T * A ...")
    indptr = io.load_pickle(osp.join(data_root, 'train_csr_indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    A = csr_matrix((np.ones(len(indices), dtype=np.float32),
                    indices - num_users, 
                    indptr[:num_users+1]),
                    shape=(num_users, num_items))
    A2 = A.T.dot(A)
    del indptr, indices, A

    print("## save...")
    io.save_pickle(osp.join(results_root, 'indptr.pkl'), A2.indptr)
    io.save_pickle(osp.join(results_root, 'indices.pkl'), A2.indices)
    io.save_pickle(osp.join(results_root, 'edge_weights.pkl'), A2.data)


if __name__ == '__main__':
    
    setproctitle.setproctitle('generate_item2item_graph-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
