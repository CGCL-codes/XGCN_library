import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from model.PPR.ppr_helper import ppr_for_batch_nodes
from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir

import numpy as np
import os.path as osp
from tqdm import tqdm
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    num_nodes = info['num_nodes']
    topk = config['topk']
    num_walks = config['num_walks']
    walk_length = config['walk_length']
    alpha = config['alpha']  # restart probability
    
    indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
    # ppr topk neighbors for each node
    nei = np.empty((num_nodes, topk), dtype=np.int64)
    # topk ppr scores
    wei = np.zeros((num_nodes, topk), dtype=np.int32)
    
    all_nids = np.arange(num_nodes)
    start = 0
    end = 0
    batch_size = 1024
    # while not finish:
    print("## run ppr ...")
    for _ in tqdm(range(int(np.ceil(num_nodes / batch_size)))):
        start = end
        end = start + batch_size
        if end > num_nodes:
            end = num_nodes
        nids = all_nids[start : end]
        batch_nei, batch_wei = ppr_for_batch_nodes(
            indptr, indices, 
            nids, 
            topk, num_walks, walk_length, alpha
        )
        nei[start : end] = batch_nei
        wei[start : end] = batch_wei
    
    print("## save ...")
    io.save_pickle(osp.join(results_root, 'nei.pkl'), nei)
    io.save_pickle(osp.join(results_root, 'wei.pkl'), wei)


if __name__ == '__main__':
    
    setproctitle.setproctitle('xgcn-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
