import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments

import torch
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    results_root = config['results_root']
    
    E = torch.load(osp.join(results_root, 'out_emb_table.pt'), map_location='cpu').detach()
    E_abs = E.abs()
    E_norm = E.norm(dim=-1)
    
    stat = {
        'E.mean()': E.mean().item(),
        'E.std()': E.std().item(),
        'E.abs.mean()': E_abs.mean().item(),
        'E.abs.std()': E_abs.std().item(),
        'E-l2norm-mean': E_norm.mean().item(),
        'E-l2norm-std': E_norm.std().item(),
    }
    
    io.save_json(osp.join(results_root, 'emb_stat.json'), stat)


if __name__ == "__main__":
    
    setproctitle.setproctitle('emb_simple_stat-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
