from ..base import BaseSampler
from XGCN.data import io, csr

import torch
import os.path as osp


class RandomNeg_Sampler(BaseSampler):
    
    def __init__(self, config, data):
        info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
        if info['graph_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
        self.num_neg = config['num_neg']
        
    def __call__(self, pos_sample_data):
        src = pos_sample_data['src']
        neg = torch.randint(self.neg_low, self.neg_high, 
                            (len(src), self.num_neg)).squeeze()
        return neg


class StrictNeg_Sampler(BaseSampler):
    
    def __init__(self, config, data):
        info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
        if info['graph_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
        self.num_neg = config['num_neg']
        
        if 'indptr' in data:
            indptr = data['indptr']
            indices = data['indices']
        else:
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            data.update({'indptr': indptr, 'indices': indices})
        self.indptr = indptr
        self.indices = indices
        
    def __call__(self, pos_sample_data):
        src = pos_sample_data['src']
        neg = []
        _set = set
        for u in src.numpy():
            nei = csr.get_neighbors(self.indptr, self.indices, u)
            nei = _set(nei)
            u_neg = []
            while len(u_neg) < self.num_neg:
                while True:
                    v = torch.randint(self.neg_low, self.neg_high, (1, )).item()
                    if v not in nei:
                        u_neg.append(v)
                        break
            neg.append(u_neg)
        neg = torch.LongTensor(neg).squeeze()
        return neg
