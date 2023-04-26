from XGCN.data import io
from XGCN.utils.parse_arguments import parse_arguments

import numpy as np


def main():
    
    config = parse_arguments()
    file_input = config['file_input']
    file_output = config['file_output']
    num_sample = config['num_sample']
    
    test_set = io.load_pickle(file_input)
    src = test_set['src']
    pos_list = test_set['pos_list']
    print("number of souce node in the test set:", len(src))
    print("num_sample:", num_sample)
    
    np.random.seed(1999)
    idx = np.arange(len(src))
    np.random.shuffle(idx)
    sampled_idx = idx[:num_sample]
    
    val_src = src[sampled_idx]
    val_pos_list = []
    pos_list = test_set['pos_list']
    for i in sampled_idx:
        val_pos_list.append(pos_list[i])

    val_set = {
        'src': val_src,
        'pos_list': val_pos_list
    }
    io.save_pickle(file_output, val_set)


if __name__ == '__main__':
    
    main()
