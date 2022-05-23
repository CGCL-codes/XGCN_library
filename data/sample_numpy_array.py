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

    X = io.load_pickle(file_input)
    
    seed = config['seed'] if 'seed' in config else 2022
    
    np.random.seed(seed)
    np.random.shuffle(X)
    
    num_sample = config['num_sample']
    X = X[:num_sample]
    print("sampled:", X.shape)
    
    io.save_pickle(file_output, X)


if __name__ == '__main__':
    
    setproctitle.setproctitle('sample_numpy_array-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
