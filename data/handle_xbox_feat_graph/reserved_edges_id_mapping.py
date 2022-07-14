import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, wc_count

import numpy as np
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def process_edges_file(file_old, dict_old2new, file_new):
    edges = []
    n = wc_count(file_old)
    with open(file_old, 'r') as f:
        for _ in range(n):
            line = f.readline()
            x = line.split()
            if x[2] == '1':
                src_old = x[0]
                dst_old = x[1]
                if src_old in dict_old2new and dst_old in dict_old2new:
                    src_new = dict_old2new[src_old]
                    dst_new = dict_old2new[dst_old]
                    edges.append((src_new, dst_new))
    edges = np.array(edges, dtype=np.int32)
    print("num edges:", len(edges))
    io.save_pickle(file_new, edges)


def main():
    
    config = parse_arguments()
    print_dict(config)

    raw_data_root = config['raw_data_root']
    data_root = config['data_root']
    
    dict_old2new = io.load_pickle(osp.join(data_root, 'dict_old2new.pkl'))
    
    process_edges_file(osp.join(raw_data_root, 'train.tsv'),
                       dict_old2new,
                       osp.join(data_root, 'reserved_train_edges.pkl'))

    process_edges_file(osp.join(raw_data_root, 'valid.tsv'),
                       dict_old2new,
                       osp.join(data_root, 'valid_edges.pkl'))

    process_edges_file(osp.join(raw_data_root, 'test.tsv'),
                       dict_old2new,
                       osp.join(data_root, 'test_edges.pkl'))


if __name__ == '__main__':
    
    setproctitle.setproctitle('id_mapping-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
