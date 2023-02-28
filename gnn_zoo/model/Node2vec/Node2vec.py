from XGCN.utils import io
from XGCN.utils import csr
from XGCN.model.base import BaseEmbeddingModel

import torch
import os.path as osp
from torch_geometric.nn import Node2Vec as pyg_Node2vec


class Node2vec(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.device = self.config['device']

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
        E_src = torch.LongTensor(E_src)
        E_dst = torch.LongTensor(E_dst)
        
        self.model = pyg_Node2vec(
            torch.cat([E_src, E_dst]).reshape(2, -1),
            embedding_dim=self.config['emb_dim'],
            walk_length=self.config['walk_length'],
            context_size=self.config['context_size'],
            walks_per_node=self.config['num_walks'],
            p=self.config['p'], q=self.config['q'],
            num_negative_samples=self.config['num_neg'],
            sparse=True
        ).to(self.device)
        
        self.opt = torch.optim.SparseAdam(self.model.parameters(), 
                                          lr=self.config['emb_lr'])
        
        pyg_node2vec_train_dl = self.model.loader(
            batch_size=self.config['train_batch_size'],
            shuffle=True, num_workers=self.config['num_workers']
        )
        self.data['pyg_node2vec_train_dl'] = pyg_node2vec_train_dl
    
    def forward_and_backward(self, batch_data):
        pos_rw, neg_rw = batch_data
        loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
    
    def on_epoch_begin(self):
        self.model.train()
    
    def on_eval_begin(self):
        self.model.eval()
        self.out_emb_table = self.model.embedding.weight.data
        
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users']:]
        else:
            self.target_emb_table = self.out_emb_table
