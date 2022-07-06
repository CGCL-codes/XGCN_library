import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict

import numpy as np
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    file_input = config['file_input']
    file_output = config['file_output']

    eval_set = io.load_pickle(file_input)
    src_list = eval_set['src']
    pos_list = eval_set['pos_list']
    
    all_idx = np.arange(len(src_list))
    seed = config['seed'] if 'seed' in config else 2022
    np.random.seed(seed)
    np.random.shuffle(all_idx)
    
    print("num all samples:", len(src_list))
    
    # num_sample = config['num_sample']
    num_sample = int(input("input num_sample="))
    print("num sample:", num_sample)
    
    sampled_idx = all_idx[:num_sample]
    
    new_src_list = []
    new_pos_list = []
    for i in sampled_idx:
        new_src_list.append(src_list[i])
        new_pos_list.append(pos_list[i])
    new_src_list = np.array(new_src_list)
    
    new_eval_set = {
        'src': new_src_list,
        'pos_list': new_pos_list
    }
    
    io.save_pickle(file_output, new_eval_set)


if __name__ == '__main__':
    
    setproctitle.setproctitle('sample_multi_pos_eval_set-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
