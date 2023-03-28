from XGCN.model.base import BaseEmbeddingModel
from XGCN.model.module import init_emb_table, dot_product
from XGCN.data import io

import torch
import torch.nn.functional as F
import os.path as osp


class UltraGCN(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.num_users = self.info['num_users']
        self.device = self.config['device']
        self.emb_table_device = self.config['emb_table_device']
        
        self.emb_table = init_emb_table(self.config, self.info['num_nodes'])
        
        self.opt_list = []
        if self.config['use_sparse']:
            self.opt_list.append(
                torch.optim.SparseAdam([{'params':list(self.emb_table.parameters()),
                                        'lr': self.config['emb_lr']}])
            )
        else:
            self.opt_list.append(
                torch.optim.Adam([{'params': self.emb_table.parameters(),
                                    'lr': self.config['emb_lr']}])
            )
        
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

    def forward_and_backward(self, batch_data):
        ((src, pos, neg), ) = batch_data
        
        src_emb = self.emb_table(src.to(self.device))
        pos_emb = self.emb_table(pos.to(self.device))
        neg_emb = self.emb_table(neg.to(self.device))
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        if self.graph_type == 'user-item':
            _pos = pos - self.num_users
            _neg = neg - self.num_users
        else:
            _pos = pos
            _neg = neg
        
        if self.config['lambda'] > 0:
            beta_pos = self.beta_uD[src] * self.beta_iD[_pos]
            beta_neg = self.beta_uD[src].unsqueeze(1) * self.beta_iD[_neg]
            pos_coe = 1 + self.config['lambda'] * beta_pos
            neg_coe = 1 + self.config['lambda'] * beta_neg
        else:
            pos_coe = None
            neg_coe = None
        
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_score, 
            torch.ones(pos_score.shape).to(self.device),
            weight=pos_coe, reduction='none'
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_score, 
            torch.zeros(neg_score.shape).to(self.device),
            weight=neg_coe, reduction='none'
        ).mean(dim = -1)
        
        loss_C_O = (pos_loss + self.config['neg_weight'] * neg_loss).sum()
        
        loss = loss_C_O
        
        # loss L_I
        if self.config['gamma'] > 0:
            ii_neighbors = self.ii_topk_neighbors[_pos]
            ii_scores = self.ii_topk_similarity_scores[_pos]
            if self.graph_type == 'user-item':
                _ii_neighbors = ii_neighbors + self.num_users
            else:
                _ii_neighbors = ii_neighbors
            ii_emb = self.emb_table(_ii_neighbors)
            
            pos_ii_score = dot_product(src_emb, ii_emb)
            loss_I = -(ii_scores * pos_ii_score.sigmoid().log()).sum()
            
            loss += self.config['gamma'] * loss_I
        
        # L2 regularization loss
        rw = self.config['L2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * ((src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum())
            if self.config['gamma'] > 0:
                L2_reg_loss += 1/2 * (ii_emb**2).sum()
            
            loss += rw * L2_reg_loss
            
        self.backward(loss)
        return loss.item()

    def backward(self, loss):
        for opt in self.opt_list:
            opt.zero_grad()
        loss.backward()
        for opt in self.opt_list:
            opt.step()

    @torch.no_grad()
    def on_eval_begin(self):
        self.out_emb_table = self.emb_table.weight
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
    