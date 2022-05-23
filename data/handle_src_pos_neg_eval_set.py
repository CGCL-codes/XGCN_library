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

    src_pos_neg = np.loadtxt(file_input, dtype=np.int32)  
    # [[src, pos, neg1, ..., negk], ... ]
    # or [[src1, pos1], [src2, pos2], ... ]
    io.save_pickle(file_output, src_pos_neg)


if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_eval_set-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
