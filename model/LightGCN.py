from utils import io
from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import dot_product, bpr_loss, ssm_loss
import torch.nn.functional as F

import torch
import dgl
import os.path as osp
from abc import abstractclassmethod


class LightGCNConv(torch.nn.Module):
    
    def __init__(self, num_layers, stack_layers=True):
        super().__init__()
        self.num_layers = num_layers
        self.stack_layers = stack_layers
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
    
    def forward(self, graph, X):
        with graph.local_scope():
            if self.stack_layers:
                X_list = [X]
                for _ in range(self.num_layers):
                    graph.srcdata['h'] = X_list[-1]
                    graph.update_all(self.gcn_msg, self.gcn_reduce)
                    X_list.append(graph.dstdata['h'])
                X_out = torch.stack(X_list, dim=1).mean(dim=1)
            else:
                for _ in range(self.num_layers):
                    graph.srcdata['h'] = X
                    graph.update_all(self.gcn_msg, self.gcn_reduce)
                    X = graph.dstdata['h']
                X_out = X
            return X_out


def get_lightgcn_out_emb(A, base_emb_table, num_gcn_layers, stack_layers=True):
    if not stack_layers:
        X = base_emb_table
        for _ in range(num_gcn_layers):
            X = A @ X
        return X
    else:
        X_list = [base_emb_table]
        for _ in range(num_gcn_layers):
            X = A @ X_list[-1]
            X_list.append(X)
        X_out = torch.stack(X_list, dim=1).mean(dim=1)
    return X_out


class BaseLightGCN(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.config = config
        self.device = config['device']
        
        self.dataset_type = self.info['dataset_type']
        if self.dataset_type == 'user-item':
            self.num_users = self.info['num_users']
        
        self.base_emb_table = init_emb_table(config, self.info['num_nodes'])
        
        self.param_list = []
        if not self.config['freeze_emb']:
            self.param_list.append({'params': self.base_emb_table,  # not use sparse
                                    'lr': config['emb_lr']})
        
        self.build_gcn(config, data)

    @abstractclassmethod
    def build_gcn(self, config, data):
        pass
    
    @abstractclassmethod
    def get_out_emb(self):
        pass
    
    def save(self, root):
        # torch.save(self.base_emb_table, osp.join(root, 'base_emb_table.pt'))
        torch.save(self.out_emb_table, osp.join(root, 'out_emb_table.pt'))
    
    def prepare_for_train(self):
        pass
    
    def prepare_for_eval(self):
        self.out_emb_table = self.get_out_emb()
        
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
    
    def __call__(self, batch_data):
        return self.forward(batch_data)
    
    def forward(self, batch_data):
        src, pos, neg = batch_data
        
        out_emb = self.get_out_emb()
        
        src_emb = out_emb[src]
        pos_emb = out_emb[pos]
        neg_emb = out_emb[neg]
        
        loss_fn_type = self.config['loss_fn']
        if loss_fn_type == 'bpr_loss':
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            
            loss = bpr_loss(pos_score, neg_score)
            
        elif loss_fn_type == 'bce_loss':
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, 
                torch.ones(pos_score.shape).to(self.device)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, 
                torch.zeros(neg_score.shape).to(self.device),
            )
            
            loss = pos_loss + self.config['neg_weight'] * neg_loss
            
        elif loss_fn_type == 'ssm_loss':
            
            loss = ssm_loss(src_emb, pos_emb, tao=self.config['tao'])
            
        else:
            assert 0
        
        rw = self.config['l2_reg_weight']
        if rw > 0:
            emb_0 = self.base_emb_table
            src_emb_0 = emb_0[src]
            pos_emb_0 = emb_0[pos]
            neg_emb_0 = emb_0[neg]
            
            if loss_fn_type == 'ssm_loss':
                L2_reg_loss = 1/2 * (1 / len(src)) * (
                    (src_emb_0**2).sum() + (pos_emb_0**2).sum()
                )
            else:
                L2_reg_loss = 1/2 * (1 / len(src)) * (
                    (src_emb_0**2).sum() + (pos_emb_0**2).sum() + (neg_emb_0**2).sum()
                )
                
            loss += rw * L2_reg_loss
        
        return loss


class LightGCN_DGL(BaseLightGCN):
    
    def __init__(self, config, data):
        super().__init__(config, data)
    
    def build_gcn(self, config, data):
        data_root = self.config['data_root']
        E_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
        E_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
        indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
        
        all_degrees = indptr[1:] - indptr[:-1]
        d_src = all_degrees[E_src]
        d_dst = all_degrees[E_dst]
        
        edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt().to(self.device)
        del indptr, all_degrees, d_src, d_dst
        
        self.undi_g = dgl.graph((E_src, E_dst)).to(self.device)
        self.undi_g.edata['ew'] = edge_weights
        
        self.gcn = LightGCNConv(num_layers=config['num_gcn_layers'], 
                                stack_layers=config['stack_layers'])
    
    def get_out_emb(self):
        return self.gcn(self.undi_g, self.base_emb_table)


class LightGCN_Torch(BaseLightGCN):
    
    def __init__(self, config, data):
        super().__init__(config, data)
            
    def build_gcn(self, config, data):
        self.Asp = io.load_pickle(osp.join(config['data_root'], 'train_csr_undi_graph.sp.pkl'))
        data['csr_graph'] = self.Asp
        # self.A = csr_helper.from_scipy_to_torch(self.Asp)
        # self.A = deepcopy(self.A).to(config['device'])
        
        undi_g = io.load_pickle(osp.join(config['data_root'], 'train_undi_graph.dgl.pkl'))
        E = undi_g.edges()
        idx = torch.stack(E)
        num_nodes = undi_g.num_nodes()
        self.A = torch.sparse_coo_tensor(
            idx, undi_g.edata['ew'], (num_nodes, num_nodes)
        ).to(config['device'])
    
    def get_out_emb(self):
        return get_lightgcn_out_emb(self.A, self.base_emb_table, self.config['num_gcn_layers'])
