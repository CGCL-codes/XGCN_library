import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir, wc_count
from utils import io

import numpy as np
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def process_word_emb_txt(file):
    '''
    return: 
        dict, str -> int, map token to int id
        X, numpy array, embedding matrix
    '''
    num_token = wc_count(file)
    X = None
    token2id = {}
    with open(file, 'r') as f:
        for i in tqdm(range(num_token)):
            line = f.readline().split()  # ['the', '0.418', '0.24968', ...]
            
            token2id[line[0]] = i
            x = np.array(list(map(float, line[1:])), dtype=np.float32)
            
            if X is None:
                dim = len(x)
                X = np.empty((num_token, dim), dtype=np.float32)
            
            X[i] = x
    return token2id, X


def main():
    
    # process pretrained word embedding (txt file)
    # https://nlp.stanford.edu/projects/glove/
    
    config = parse_arguments()
    print_dict(config)
    
    file_input = config['file_input']
    results_root = config['results_root']  # place to save token2id dict and embeddings
    ensure_dir(results_root)
    
    token2id, X = process_word_emb_txt(file_input)
    
    io.save_pickle(osp.join(results_root, 'dict_token2id.pkl'), token2id)
    io.save_pickle(osp.join(results_root, 'word_embeddings.pkl'), X)


if __name__ == '__main__':
    
    setproctitle.setproctitle('process_txt_word_emb-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
