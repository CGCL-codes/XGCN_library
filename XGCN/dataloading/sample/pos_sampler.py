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


class NodeBased_ObservedEdges_Sampler(BaseSampler):
    
    def __init__(self, config, data):
        # ensure sample indices:
        assert 'str_num_total_samples' in config and config['str_num_total_samples'] in ['num_nodes', 'num_users']
        
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
        self.info = io.load_yaml(osp.join(config['data_root'], 'info.yaml'))
    
    def __call__(self, nid):
        src = nid
        pos = []
        for u in src.numpy():
            nei = csr.get_neighbors(self.indptr, self.indices, u)
            if len(nei) == 0:
                pos.append(src)
                continue
            else:
                idx = torch.randint(0, len(nei), (1,)).item()
                pos.append(nei[idx])
        pos = torch.LongTensor(pos)
        return src, pos
