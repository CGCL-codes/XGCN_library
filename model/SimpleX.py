from utils import io
from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import cosine_contrastive_loss

import torch
import dgl
import dgl.dataloading as dgldl
import os.path as osp


class SimpleX(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.config = config
        self.device = config['device']
        
        self.dataset_type = self.info['dataset_type']
        if self.dataset_type == 'user-item':
            self.num_users = self.info['num_users']
        
        self.base_emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = torch.empty(self.base_emb_table.weight.shape, dtype=torch.float32)

        self.param_list = {'SparseAdam': []}
        self.param_list['SparseAdam'].append({
            'params': list(self.base_emb_table.parameters()), 
            'lr': config['emb_lr']
        })

        data_root = self.config['data_root']
        
        if self.dataset_type == 'user-item':
            # reverse graph
            E_dst = io.load_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'))
            E_src = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
        else:
            # undirected graph
            E_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
            E_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
        self.g = dgl.graph((E_src, E_dst)).to(self.device)
        
        self.node_collator = dgldl.NodeCollator(
            self.g,
            self.g.nodes(),
            dgldl.MultiLayerFullNeighborSampler(1)
        )

        if self.config['use_uniform_weight']:
            self.fn_msg = dgl.function.copy_u('h', 'm')
            self.fn_reduce = dgl.function.mean(msg='m', out='h')  # average pooling
        else:
            print("## use lightgcn edge weights")
            indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
            all_degrees = indptr[1:] - indptr[:-1]
            d_src = all_degrees[E_src]
            d_dst = all_degrees[E_dst]
            edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()
            self.g.edata['ew'] = edge_weights.to(self.device)
            self.fn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
            self.fn_reduce = dgl.function.sum(msg='m', out='h')
    
    def _get_user_output_emb(self, users=None):
        if users is None:
            with self.g.local_scope():
                self.g.srcdata['h'] = self.base_emb_table.weight
                self.g.update_all(self.fn_msg, self.fn_reduce)
                if self.dataset_type == 'user-item':
                    aggregated_item_emb = self.g.dstdata['h'][:self.num_users]
                else:
                    aggregated_item_emb = self.g.dstdata['h']
            
            if self.dataset_type == 'user-item':
                user_self_emb = self.base_emb_table.weight[:self.num_users]
            else:
                user_self_emb = self.base_emb_table.weight
        else:
            input_items, _, blocks = self.node_collator.collate(users.to(self.device))
            block = blocks[0]
            
            with block.local_scope():
                block.srcdata['h'] = self.base_emb_table(input_items.to(self.device))
                block.update_all(self.fn_msg, self.fn_reduce)
                aggregated_item_emb = block.dstdata['h']
            
            user_self_emb = self.base_emb_table(users.to(self.device))
            
        theta = self.config['theta']
        user_output_emb = theta * user_self_emb + (1 - theta) * aggregated_item_emb
        
        return user_output_emb
        
    def __call__(self, batch_data):
        return self.forward(batch_data)
        
    def forward(self, batch_data):
        src, pos, neg = batch_data
        
        src_emb = self._get_user_output_emb(src.to(self.device))
        pos_emb = self.base_emb_table(pos.to(self.device))
        neg_emb = self.base_emb_table(neg.to(self.device))
        
        loss = cosine_contrastive_loss(src_emb, pos_emb, neg_emb, 
                                       self.config['margin'], self.config['neg_weight'])
        
        if self.config['l2_reg_weight'] > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * ((src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum())
            loss += self.config['l2_reg_weight'] * L2_reg_loss
        
        return loss

    def prepare_for_train(self):
        pass
    
    def prepare_for_eval(self):
        if self.dataset_type == 'user-item':
            self.out_emb_table[:self.num_users] = self._get_user_output_emb().cpu()
            self.out_emb_table[self.num_users:] = self.base_emb_table.weight[self.num_users:].cpu()
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.out_emb_table = self.base_emb_table.weight
            self.target_emb_table = self.out_emb_table
        
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
