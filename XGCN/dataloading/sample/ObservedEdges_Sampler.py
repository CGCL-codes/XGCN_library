from ..base import BaseSampler
from XGCN.utils import io
from XGCN.utils import csr

import torch
import os.path as osp


def build_ObservedEdges_Sampler(config, data):
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
    
    sampler = ObservedEdges_Sampler(E_src, E_dst)
    return sampler


class ObservedEdges_Sampler(BaseSampler):
    
    def __init__(self, E_src, E_dst):
        self.E_src = E_src
        self.E_dst = E_dst
        
    def __call__(self, eid):
        src = self.E_src[eid]
        pos = self.E_dst[eid]
        return src, pos
