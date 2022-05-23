import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict

import numpy as np
import setproctitle
import time


def handle_adj_eval_data(filename):
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
    print_dict(config)
    
    file_input = config['file_input']
    file_output = config['file_output']
    
    eval_set = handle_adj_eval_data(file_input)
    io.save_pickle(file_output, eval_set)


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_adj_eval_set-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
