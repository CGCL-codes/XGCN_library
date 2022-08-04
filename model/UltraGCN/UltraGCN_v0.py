from utils import io
from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import dot_product
from model.module import ssm_loss

import torch
import torch.nn.functional as F
import os.path as osp


class UltraGCN_v0(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.config = config
        self.device = config['device']
        
        self.base_emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = None
        self.target_emb_table = None
        
        self.param_list = {'Adam': []}
        self.param_list['Adam'].append({
            'params': list(self.base_emb_table.parameters()), 
            'lr': config['emb_lr']
        })
        
        if self.config['loss_fn'] != 'ssm_loss':
            if self.config['lambda'] > 0:
                constrain_mat = io.load_pickle(config['file_ultra_constrain_mat'])
                self.beta_uD = torch.FloatTensor(constrain_mat['beta_users']).to(self.device)
                self.beta_iD = torch.FloatTensor(constrain_mat['beta_items']).to(self.device)
            
            if self.config['gamma'] > 0:
                self.ii_topk_neighbors = io.load_pickle(config['file_ii_topk_neighbors'])
                self.ii_topk_similarity_scores = io.load_pickle(config['file_ii_topk_similarity_scores'])
                
                topk = config['topk']
                self.ii_topk_neighbors = torch.LongTensor(self.ii_topk_neighbors[:, :topk]).to(self.device)
                self.ii_topk_similarity_scores = torch.FloatTensor(self.ii_topk_similarity_scores[:, :topk]).to(self.device)
        
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, 'out_emb_table.pt'))
        
    def prepare_for_train(self):
        pass
    
    def prepare_for_eval(self):
        self.out_emb_table = self.base_emb_table.weight
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
    
    def __call__(self, batch_data):
        return self.forward(batch_data)
    
    def forward(self, batch_data):
        src, pos, neg = batch_data
        
        src_emb = self.base_emb_table(src.to(self.device))
        pos_emb = self.base_emb_table(pos.to(self.device))
        neg_emb = self.base_emb_table(neg.to(self.device))
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        log_pos = (pos_score.sigmoid() + 1e-8).log()
        log_neg = ((-neg_score).sigmoid() + 1e-8).log()
        
        if self.dataset_type == 'user-item':
            _pos = pos - self.num_users
            _neg = neg - self.num_users
        else:
            _pos = pos
            _neg = neg
        beta_pos = self.beta_uD[src] * self.beta_iD[_pos]
        beta_neg = self.beta_uD[src].unsqueeze(1) * self.beta_iD[_neg]
        
        # L = -(w1 + w2*\beta)) * log(sigmoid(e_u e_i)) - \sum_{N-} (w3 + w4*\beta) * log(sigmoid(- e_u e_i'))
        pos_loss = ((self.config['w1'] + self.config['w2'] * beta_pos) * log_pos)
        neg_loss = ((self.config['w3'] + self.config['w4'] * beta_neg) * log_neg).mean(dim=-1)
        loss = -(pos_loss + self.config['neg_weight'] * neg_loss).sum()
        
        ii_neighbors = self.ii_topk_neighbors[_pos]
        ii_scores = self.ii_topk_similarity_scores[_pos]
        if self.dataset_type == 'user-item':
            _ii_neighbors = ii_neighbors + self.num_users
        else:
            _ii_neighbors = ii_neighbors
        ii_emb = self.base_emb_table(_ii_neighbors.to(self.device))
        
        pos_ii_score = dot_product(src_emb, ii_emb)
        loss_I = -(ii_scores * pos_ii_score.sigmoid().log()).sum()
        
        loss += self.config['gamma'] * loss_I
        
        loss += self.config['l2_reg_weight'] * 0.5 * torch.sum(self.base_emb_table.weight ** 2)
        
        return loss
