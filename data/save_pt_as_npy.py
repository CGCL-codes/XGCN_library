import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments

import numpy as np
import torch
import setproctitle
import time


def main():
    
    config = parse_arguments()
    
    X = torch.load(config['file_input'], map_location='cpu')
    X = X.detach().numpy()
    np.save(config['file_output'], X)


if __name__ == '__main__':
    
    setproctitle.setproctitle('save_pt_as_npy-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
