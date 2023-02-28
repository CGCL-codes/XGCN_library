from gnn_zoo.model.LightGCN import get_lightgcn_out_emb
from gnn_zoo.utils import io
from gnn_zoo.utils import csr

import numpy as np
import torch
import os.path as osp


class LightGCN_Propagation:
    
    def __init__(self, config, data):
        self.config = config
        self.data = data
        
        data_root = self.config['data_root']
        
        if 'undi_indptr' in self.data:
            indptr = self.data['undi_indptr']
            indices = self.data['undi_indptr']
        else:
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            indptr, indices = csr.get_undirected(indptr, indices)
        
        E_src = csr.get_src_indices(indptr)
        E_dst = indices
        
        self.num_nodes = len(indptr) - 1
        all_degrees = csr.get_degrees(indptr)
        
        d_src = all_degrees[E_src]
        d_dst = all_degrees[E_dst]
        
        edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()
        del d_src, d_dst
            
        if hasattr(torch, 'sparse_csr_tensor'):
            print("## use torch.sparse_csr_tensor")
            self.A = torch.sparse_csr_tensor(
                torch.LongTensor(np.array(indptr, dtype=np.int64)),
                torch.LongTensor(E_dst),
                edge_weights,
                (self.num_nodes, self.num_nodes)
            )
        else:
            print("## use torch.sparse_coo_tensor")
            del indptr
            E_src = torch.IntTensor(E_src)
            E_dst = torch.IntTensor(E_dst)
            E = torch.cat([E_src, E_dst]).reshape(2, -1)
            del E_src, E_dst
            self.A = torch.sparse_coo_tensor(
                E, edge_weights, 
                (self.num_nodes, self.num_nodes)
            )
    
    def __call__(self, X):
        X_out = get_lightgcn_out_emb(
            self.A, X.cpu(), self.config['num_gcn_layers'],
            stack_layers=self.config['stack_layers']
        )
        return X_out
