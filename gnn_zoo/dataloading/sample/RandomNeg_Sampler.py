from ..base import BaseSampler
from XGCN.utils import io

import torch
import os.path as osp


def build_RandomNeg_Sampler(config, data):
    data_root = config['data_root']
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    
    sampler = RandomNeg_Sampler(info, config['num_neg'])
    return sampler

    
class RandomNeg_Sampler(BaseSampler):
    
    def __init__(self, info, num_neg):
        if info['graph_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
        
        self.num_neg = num_neg
        
    def __call__(self, pos_sample_data):
        src = pos_sample_data['src']
        neg = torch.randint(self.neg_low, self.neg_high, 
                            (len(src), self.num_neg)).squeeze()
        return neg
