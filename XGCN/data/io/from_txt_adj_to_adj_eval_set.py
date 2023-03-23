from XGCN.utils import io
from XGCN.utils.parse_arguments import parse_arguments

import numpy as np


def from_txt_adj_to_adj_eval_set(filename):
    src = []
    pos_list = []
    with open(filename, 'r') as f:
        while True:
            x = np.loadtxt(f, max_rows=1, dtype=np.int32, ndmin=1)
            if len(x) == 0:
                break
            if len(x) == 1:
                continue
            src.append(x[0])
            pos_list.append(x[1:])
    print("(<-- don't mind if any UserWarning pops up)")
    src = np.array(src, dtype=np.int32)
    eval_set = {'src': src, 'pos_list': pos_list}
    return eval_set


def main():
    
    config = parse_arguments()
    
    file_input = config['file_input']
    file_output = config['file_output']
    
    eval_set = from_txt_adj_to_adj_eval_set(file_input)
    io.save_pickle(file_output, eval_set)


if __name__ == '__main__':
    
    main()
