from ..base import BaseSampler
from XGCN.data import io, csr

import torch
import os.path as osp


class ObservedEdges_Sampler(BaseSampler):
    
    def __init__(self, config, data):
        if 'indptr' in data:
            indptr = data['indptr']
            indices = data['indices']
        else:
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            data.update({'indptr': indptr, 'indices': indices})
        src_indices = csr.get_src_indices(indptr)
        E_src = torch.LongTensor(src_indices)
        E_dst = torch.LongTensor(indices)
        self.E_src = E_src
        self.E_dst = E_dst
        
    def __call__(self, eid):
        src = self.E_src[eid]
        pos = self.E_dst[eid]
        return src, pos
