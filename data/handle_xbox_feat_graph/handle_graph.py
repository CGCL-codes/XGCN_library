import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir, wc_count, ReIndexDict
from data.handle_train_graph import handle_train_graph

import numpy as np
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    ensure_dir(data_root)
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    assert dataset_type in ['user-item', 'social']
    
    file_input = config['file_input']
    
    num_edges = wc_count(file_input)
    
    E_src = np.empty(num_edges, dtype=np.int32)
    E_dst = np.empty(num_edges, dtype=np.int32)
    
    dic = ReIndexDict()  # re index, map old id to new id
    
    print("load and reindex from .txt")
    with open(file_input, 'r') as f:
        for i in tqdm(range(num_edges)):
            line = f.readline()
            x = line.split()
            E_src[i] = dic[x[0]]
            E_dst[i] = dic[x[1]]
    
    io.save_pickle(osp.join(data_root, 'dict_old2new.pkl'), dic.get_old2new_dict())
    io.save_pickle(osp.join(data_root, 'dict_new2old.pkl'), dic.get_new2old_dict())
    del dic
    
    handle_train_graph(E_src, E_dst, data_root, dataset_name, dataset_type)


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_graph-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
