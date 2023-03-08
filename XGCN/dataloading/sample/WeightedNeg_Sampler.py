from XGCN.dataloading.base import BaseSampler
from XGCN.utils import io, csr

import torch
import os.path as osp


class WeightedNeg_Sampler(BaseSampler):
    
    def __init__(self, config, data):
        self.num_neg = config['num_neg']
        
        data_root = config['data_root']
        indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
        indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
        indptr, indices = csr.get_undirected(indptr, indices)
        degrees = indptr[1:] - indptr[:-1]
        
        info = io.load_yaml(osp.join(data_root, 'info.yaml'))
        if info['graph_type'] == 'user-item':
            self.num_neg_total = info['num_items']
            self.offset = info['num_users']
        else:
            self.num_neg_total = info['num_nodes']
            self.offset = 0
        
        # the probability a node is sampled is proportional to the weights:
        self.weights = torch.FloatTensor(
            degrees[self.offset : self.offset + self.num_neg_total]
        ) ** 0.75
        
    def __call__(self, pos_sample_data):
        src = pos_sample_data['src']
        neg = torch.multinomial(
            self.weights, num_samples=len(src), replacement=True
        ) + self.offset
        return neg
