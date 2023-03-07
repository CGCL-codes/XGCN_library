from XGCN.utils import io
from XGCN.utils.parse_arguments import parse_arguments

import numpy as np


def main():
    
    config = parse_arguments()
    
    file_input = config['file_input']
    file_output = config['file_output']

    X = np.loadtxt(file_input, dtype=np.int32)
    io.save_pickle(file_output, X)


if __name__ == '__main__':
    
    main()
