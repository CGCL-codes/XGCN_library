import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments

import numpy as np
import torch
import dgl
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    data_root = config['data_root']
    results_root = config['results_root']
    device = config['device']
    
    out_emb_table = torch.load(osp.join(results_root, 'out_emb_table.pt'), map_location=device).detach()
    
    E_src = io.load_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'))
    E_dst = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    g = dgl.graph((E_src, E_dst))
    in_degrees = g.in_degrees()
    
    # val_edges = io.load_pickle(osp.join(data_root, 'val_edges-1000.pkl'))
    # src = val_edges[:,0]
    
    all_nids = np.arange(g.num_nodes())
    np.random.seed(2022)
    np.random.shuffle(all_nids)
    src = all_nids[:1000]
    
    k = 100
    top_reco_indegrees = np.empty((len(src), k), dtype=np.int32)
    
    for i in tqdm(range(len(src))):
        u = src[i]
        src_emb = out_emb_table[u]
        all_scores = src_emb @ out_emb_table.t()
        _, top100_nid = all_scores.topk(k)
        top_reco_indegrees[i] = in_degrees[top100_nid].numpy()
    
    mean = top_reco_indegrees.mean()
    std = top_reco_indegrees.std()
    print("top_reco_indegrees.mean(): ", mean)
    print("top_reco_indegrees.std(): ", std)
    
    io.save_pickle(osp.join(results_root, 'top_reco_indegrees.pkl'), top_reco_indegrees)
    io.save_json(osp.join(results_root, 'top_reco_indegrees.json'), {
        'top_reco_indegrees.mean()': mean,
        'top_reco_indegrees.std()': std,
    })


if __name__ == "__main__":
    
    setproctitle.setproctitle('top_reco_indegree-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
