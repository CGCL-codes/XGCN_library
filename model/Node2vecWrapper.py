from utils import io
from model.BaseEmbeddingModel import BaseEmbeddingModel

import torch
import os.path as osp
from torch_geometric.nn import Node2Vec


class Node2vecWrapper(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.device = self.config['device']
        
        self.dataset_type = self.info['dataset_type']
        if self.dataset_type == 'user-item':
            self.num_users = self.info['num_users']
        self.num_nodes = self.info['num_nodes']
        
        data_root = self.config['data_root']
        E_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
        E_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
        
        E_src = torch.LongTensor(E_src)
        E_dst = torch.LongTensor(E_dst)
        
        self.model = Node2Vec(
            torch.cat([E_src, E_dst]).reshape(2, -1),
            embedding_dim=config['emb_dim'],
            walk_length=config['walk_length'],
            context_size=config['context_size'],
            walks_per_node=config['num_walks'],
            p=config['p'], q=config['q'],
            num_negative_samples=config['num_neg'],
            sparse=True
        ).to(self.device)
        
    def __call__(self, batch_data):
        return self.forward(batch_data)
    
    def forward(self, batch_data):
        pos_rw, neg_rw = batch_data
        loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
        return loss
    
    def prepare_for_train(self):
        self.model.train()
    
    def prepare_for_eval(self):
        self.model.eval()
        self.out_emb_table = self.model.embedding.weight.data
        
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
        
        # _abs = self.out_emb_table.abs()
        # print("## emb_table.abs().mean():", _abs.mean())
        # print("## emb_table.abs().std():", _abs.std())
        
    def save(self, root, file_out_emb_table=None):
        if file_out_emb_table is None:
            file_out_emb_table = "out_emb_table.pt"
        torch.save(self.out_emb_table, osp.join(root, file_out_emb_table))
