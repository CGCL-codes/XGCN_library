import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict

import numpy as np
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    file_input = config['file_input']
    file_output = config['file_output']
    num_neg = config['num_neg']
    
    info = io.load_yaml(osp.join(osp.join(data_root, 'info.yaml')))
    pos_edges = io.load_pickle(file_input)
    
    if isinstance(pos_edges, dict):
        src_list = pos_edges['src']
        pos_list = pos_edges['pos_list']
        E_src = []
        E_pos = []
        for i in range(len(src_list)):
            src = src_list[i]
            pos = pos_list[i]
            for _ in range(len(pos)):
                E_src.append(src)
            E_pos.append(pos)
        E_src = np.array(E_src)
        E_pos = np.concatenate(E_pos)
        pos_edges = np.stack([E_src, E_pos]).transpose()
    
    seed = config['seed'] if 'seed' in config else 2022
    
    np.random.seed(seed)
    low = 0
    if info['dataset_type'] == 'user-item':
        high = info['num_item']
    else:
        high = info['num_nodes']
    neg = np.random.randint(low, high, (len(pos_edges), num_neg))
    
    src_pos_neg = np.concatenate([pos_edges, neg], axis=-1)
    print(src_pos_neg)
    print(src_pos_neg.shape)
    io.save_pickle(file_output, src_pos_neg)


if __name__ == '__main__':
    
    setproctitle.setproctitle('sample_neg-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
