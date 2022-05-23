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
    file_output_2 = config['file_output_2']
    
    # input: .txt file
    # ouput: numpy array, saved using pickle
    
    # input format: src \t dst \t label
    # each src has 1 pos and k neg
    # [[src01, dst01, 1], 
    #  [src01, dst02, 0], 
    #   ..., 
    #  [src01, dst0k, 0],
    #  [src02, dst01, 1], 
    #  [src02, dst02, 0], 
    #   ..., 
    #  [src02, dst0k, 0], ...]
    X = np.loadtxt(file_input, dtype=np.int32, delimiter='\t')
    
    i = 1
    while X[i, 2] == 0:
        i += 1
    num_neg = i - 1
    num_src = int(len(X) / (1 + num_neg))
    
    print("## num_src:", num_src)
    print("## num_neg per pos:", num_neg)
    
    src_pos_neg = np.empty(shape=(num_src, 1 + 1 + num_neg),dtype=int)
    
    for i in range(num_src):
        start = (1 + num_neg) * i
        pos_neg = X[start:start + 1 + num_neg, 1]
        
        src_pos_neg[i][0] = X[start, 0]
        src_pos_neg[i][1:] = pos_neg
    
    io.save_pickle(file_output, src_pos_neg)
    io.save_pickle(file_output_2, src_pos_neg[:, :2])  # pos edges


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_labelled_eval_set-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
