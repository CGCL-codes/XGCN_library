import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict
from data.handle_train_graph import handle_train_graph

import numpy as np
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    dataset_name = config['dataset_name']
    dataset_type = config['dataset_type']
    assert dataset_type in ['user-item', 'social']
    
    file_input = config['file_input']
    
    print("## load .txt edge list ...")
    E = np.loadtxt(fname=file_input, dtype=np.int32)
    E_src = E[:, 0]
    E_dst = E[:, 1]
    
    handle_train_graph(E_src, E_dst, data_root, dataset_name, dataset_type)


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_edge_list_txt-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
