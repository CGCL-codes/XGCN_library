import XGCN
from XGCN.data import io
from XGCN.utils.parse_arguments import parse_arguments

import numpy as np


def main():
    
    print("# process_evaluation_set...")
    
    config = parse_arguments()
    file_input = config['file_input']
    file_output = config['file_output']
    
    eval_method = config['eval_method']
    assert eval_method in [
        'one_pos_k_neg',
        'one_pos_whole_graph',
        'multi_pos_whole_graph'
    ]
    
    if eval_method in ['one_pos_k_neg', 'one_pos_whole_graph']:
        eval_set = np.loadtxt(fname=file_input, dtype=np.int32)
    elif eval_method == 'multi_pos_whole_graph':
        eval_set = io.from_txt_adj_to_adj_eval_set(file_input)    
    else:
        assert 0
    
    io.save_pickle(file_output, eval_set)
    print("# done!")


if __name__ == '__main__':
    main()
