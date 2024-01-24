from XGCN.model.base import BaseEmbeddingModel
from XGCN.model.module import init_emb_table, dot_product, bpr_loss, bce_loss
from XGCN.data import io, csr
from XGCN.utils.utils import gram_schmidt

import numpy as np
import torch
import os.path as osp
from tqdm import tqdm


class RandNE(BaseEmbeddingModel):
    
    def __init__(self, config):
        super().__init__(config)

        if 'undi_indptr' in self.data:
            indptr = self.data['undi_indptr']
            indices = self.data['undi_indptr']
        else:
            data_root = self.config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            indptr, indices = csr.get_undirected(indptr, indices)
        
        E_src = csr.get_src_indices(indptr)
        E_dst = indices
        
        self.num_nodes = len(indptr) - 1

        if self.config['use_lightgcn_coe']:
            all_degrees = csr.get_degrees(indptr)
            d_src = all_degrees[E_src]
            d_dst = all_degrees[E_dst]
            edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()
            del d_src, d_dst
        else:
            edge_weights = torch.ones(len(E_src))

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
        
        self.U0 = init_emb_table(
            self.config, self.info['num_nodes'], return_tensor=True
        )
        self.out_emb_table = None

        if self.config['orthogonal']:
            print("using gram_schmidt to make columns orthogonal")
            self.U0 = gram_schmidt(self.U0.T).T

    def fit(self):
        Us = [self.U0]
        for i in range(self.config['num_gcn_layers']):
            U = torch.sparse.mm(self.A, Us[i])
            Us.append(U)
        self.out_emb_table = torch.zeros(size=self.U0.shape)

        alpha_list = eval(self.config['alpha_list'])
        for i, U in enumerate(Us):
            print("propagation {}/{}".format(i + 1, len(Us)))
            self.out_emb_table += alpha_list[i] * U

        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table

    def save(self, root=None):
        self._save_out_emb_table(root)
    
    def load(self, root=None):
        self._load_out_emb_table(root)
