import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, wc_count
from data.handle_train_graph import handle_train_graph

import numpy as np
from tqdm import tqdm
import setproctitle
import time


def load_adj_as_edge_list(filename):
    num_lines = wc_count(filename)
    src_list = []
    dst_list = []
    with open(filename, 'r') as f:
        for _ in tqdm(range(num_lines)):
            # line = f.readline()  # src \t dst01,dst02,... 
            # src, dst = line.split()
            
            # dst = np.array(list(map(lambda x:int(x), dst.split(','))),
            #                dtype=np.int32)
            # src = np.full(len(dst), int(src), dtype=np.int32)
            
            line = np.loadtxt(f, dtype=np.int32, max_rows=1)
            if len(line) < 2:
                continue
            dst = line[1:]
            src = np.full(len(dst), line[0], dtype=np.int32)
            
            src_list.append(src)
            dst_list.append(dst)
    E_src = np.concatenate(src_list, dtype=np.int32)
    E_dst = np.concatenate(dst_list, dtype=np.int32)
    return E_src, E_dst


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']  # place to save the processed data
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    assert dataset_type in ['user-item', 'social']
    
    file_input = config['file_input']
    
    print("## load .txt adj graph", file_input)
    E_src, E_dst = load_adj_as_edge_list(file_input)
   
    handle_train_graph(E_src, E_dst, data_root, dataset_name, dataset_type)


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_adj_graph_txt-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
