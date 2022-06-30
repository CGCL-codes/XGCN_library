import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from model.PPR.ppr_helper import ppr_for_batch_nodes_on_whole_graph
from model.BaseEmbeddingModel import _mask_neighbor_score

import numpy as np
import torch
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    indptr = io.load_pickle(osp.join(data_root, 'train_csr_indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    undi_indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
    undi_indices = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    
    all_nids = np.arange(info['num_nodes'])
    np.random.seed(2022)
    np.random.shuffle(all_nids)
    case_study_nids = all_nids[:config['num_sample']]
    dl = torch.utils.data.DataLoader(case_study_nids, batch_size=128)
    
    topk = config['topk']
    R = np.empty(shape=(len(case_study_nids), topk), dtype=np.int64)
    st = 0
    for src in tqdm(dl):
        src = src.numpy()
        all_target_score = ppr_for_batch_nodes_on_whole_graph(
            indptr=undi_indptr, 
            indices=undi_indices, 
            nids=src, 
            num_walks=config['num_walks'], 
            walk_length=config['walk_length'], 
            alpha=config['alpha']
        )
        _mask_neighbor_score(indptr, indices, src, all_target_score)
        
        _, batch_R = torch.tensor(all_target_score).topk(topk)
        
        R[st : st + len(batch_R)] = batch_R.numpy()
        st += len(batch_R)

    import pdb; pdb.set_trace()
    io.save_pickle(osp.join(results_root, 'top-reco.pkl'), R)


if __name__ == "__main__":
    
    setproctitle.setproctitle('get_ppr_top_reco-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
