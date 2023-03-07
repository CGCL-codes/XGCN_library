from XGCN.model.PPR.ppr_helper import ppr_for_batch_nodes
from XGCN.utils import io
from XGCN.utils import csr
from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import print_dict, ensure_dir, set_random_seed

import numpy as np
import os.path as osp
from tqdm import tqdm


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    seed = config['seed'] if 'seed' in config else 1999
    set_random_seed(seed)
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    num_nodes = info['num_nodes']
    topk = config['topk']
    num_walks = config['num_walks']
    walk_length = config['walk_length']
    alpha = config['alpha']  # restart probability
    
    indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
    
    indptr, indices = csr.get_undirected(indptr, indices)  # run ppr on the undirected graph
    
    # ppr topk neighbors for each node
    nei = np.empty((num_nodes, topk), dtype=np.int64)
    # topk ppr scores
    wei = np.zeros((num_nodes, topk), dtype=np.int32)
    
    all_nids = np.arange(num_nodes)
    start = 0
    end = 0
    batch_size = 1024
    
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
    
    main()
