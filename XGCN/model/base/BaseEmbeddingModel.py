from .BaseModel import BaseModel
from XGCN.model.module import dot_product
from XGCN.model.module.mask_neighbor_score import mask_neighbor_score, mask_neighbor_score_user_item
from XGCN.data import io

import torch
import numpy as np
import os.path as osp


class BaseEmbeddingModel(BaseModel):
    
    def __init__(self, config, data):
        self.config = config
        self.data = data
        
        self.data_root = self.config['data_root']
        self.results_root = self.config['results_root']
        
        self.info = io.load_yaml(osp.join(self.data_root, 'info.yaml'))
        self.graph_type = self.info['graph_type']
        if self.graph_type == 'user-item':
            self.num_users = self.info['num_users']
        
        self.indptr = None
        self.indices = None
    
    def eval(self, batch_data, eval_type):
        return {
            'whole_graph_multi_pos': self.eval_whole_graph_multi_pos,
            'whole_graph_one_pos': self.eval_whole_graph_one_pos,
            'one_pos_k_neg': self.eval_one_pos_k_neg
        }[eval_type](batch_data)
        
    def save(self, root=None):
        if root is None:
            root = self.results_root
        torch.save(self.out_emb_table, osp.join(root, 'out_emb_table.pt'))
    
    def load(self, root=None):
        if root is None:
            root = self.results_root
        self.out_emb_table = torch.load(osp.join(root, 'out_emb_table.pt'))
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users']:]
        else:
            self.target_emb_table = self.out_emb_table
    
    @torch.no_grad()
    def eval_whole_graph_multi_pos(self, batch_data):
        src, _ = batch_data
        
        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        
        return all_target_score
    
    @torch.no_grad()
    def eval_whole_graph_one_pos(self, batch_data):
        src, pos = batch_data

        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        
        pos_score = np.empty((len(src),), dtype=np.float32)
        for i in range(len(src)):
            pos_score[i] = all_target_score[i][pos[i]]
        pos_neg_score = np.concatenate((pos_score.reshape(-1, 1), all_target_score), axis=-1)
        
        return pos_neg_score
    
    @torch.no_grad()
    def eval_one_pos_k_neg(self, batch_data):
        src, pos, neg = batch_data
        
        src_emb = self.out_emb_table[src]
        pos_emb = self.target_emb_table[pos]
        neg_emb = self.target_emb_table[neg]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        pos_neg_score = torch.cat((pos_score.view(-1, 1), neg_score), dim=-1).cpu().numpy()
        return pos_neg_score
    
    def infer_all_target_score(self, src, mask_nei=True):
        src_emb = self.out_emb_table[src]
        
        all_target_score = (src_emb @ self.target_emb_table.t()).cpu().numpy()
        
        if mask_nei:
            self.mask_neighbor_score(src, all_target_score)
        
        return all_target_score
        
    def mask_neighbor_score(self, src, all_target_score):
        if self.indptr is None:
            self.prepare_train_graph_for_mask()
            
        if self.graph_type == 'user-item':
            mask_neighbor_score_user_item(self.indptr, self.indices,
                src, all_target_score, self.num_users
            )
        else:
            mask_neighbor_score(self.indptr, self.indices,
                src, all_target_score
            )
    
    def prepare_train_graph_for_mask(self):
        if 'indptr' in self.data:
            self.indptr = self.data['indptr']
            self.indices = self.data['indices']
        else:
            self.indptr = io.load_pickle(osp.join(self.data_root, 'indptr.pkl'))
            self.indices = io.load_pickle(osp.join(self.data_root, 'indices.pkl'))

    def save_emb_as_txt(self, filename='out_emb_table.txt', fmt='%.6f'):
        np.savetxt(fname=filename, X=self.out_emb_table.cpu().numpy(), fmt=fmt)

    def infer_target_score(self, src, target):
        src_emb = self.out_emb_table[src]
        target_emb = self.out_emb_table[target]
        target_score = dot_product(src_emb, target_emb).cpu().numpy()
        return target_score
    
    def infer_topk(self, k, src, mask_nei=True):
        all_target_score = self.infer_all_target_score(src, mask_nei)
        score, node = torch.topk(all_target_score, k, dim=-1)
        return score, node
