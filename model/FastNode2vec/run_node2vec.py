import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from model.FastNode2vec import FastNode2vec

import torch
import os.path as osp
import time
import setproctitle


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, "config.yaml"), config)
    
    indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
    
    model = FastNode2vec(indptr, indices)
    
    try:
        model.run_node2vec(
            dim=config['emb_dim'], 
            epochs=config['epochs'], 
            alpha=config['emb_lr'],
            min_alpha=config['emb_lr'],
            num_walks=config['num_walks'], 
            walk_length=config['walk_length'], 
            window=config['context_size'], 
            p=config['p'], 
            q=config['q'],
        )
    except KeyboardInterrupt:
        pass
    
    embs = torch.FloatTensor(model.get_embeddings())
    torch.save(embs, osp.join(results_root, 'out_emb_table.pt'))


if __name__ == "__main__":
    
    setproctitle.setproctitle('n2v-' + 
                              time.strftime("%m%d-%H%M%S", time.localtime(time.time())))    
    main()
