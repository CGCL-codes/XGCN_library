from XGCN.data import io, csr
from XGCN.model.base import BaseEmbeddingModel
from XGCN.model.module import init_emb_table, cosine_contrastive_loss

import torch
import dgl
import dgl.dataloading as dgldl
import os.path as osp


class SimpleX(BaseEmbeddingModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.device = self.config['device']
        
        self.emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = torch.empty(self.emb_table.weight.shape, dtype=torch.float32)

        data_root = self.config['data_root']
        indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
        indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
        if self.graph_type == 'user-item':
            # reverse graph
            E_src = indices
            E_dst = csr.get_src_indices(indptr)
        else:
            # undirected graph
            undi_indptr, undi_indices = csr.get_undirected(indptr, indices)
            E_src = csr.get_src_indices(undi_indptr)
            E_dst = undi_indices
        self.g = dgl.graph((E_src, E_dst)).to(self.device)
        
        if self.config['train_num_layer_sample'] == '[]':
            block_sampler = dgldl.MultiLayerFullNeighborSampler(1)
        else:
            layer_sample = eval(self.config['train_num_layer_sample'])
            assert len(layer_sample) == 1
            block_sampler = dgldl.MultiLayerNeighborSampler(layer_sample)
        self.node_collator = dgldl.NodeCollator(
            self.g, self.g.nodes(), block_sampler
        )
        
        if self.config['use_uniform_weight']:
            self.fn_msg = dgl.function.copy_u('h', 'm')
            self.fn_reduce = dgl.function.mean(msg='m', out='h')  # average pooling
        else:
            print("## use lightgcn edge weights")
            all_degrees = self.g.out_degrees()
            E_src, E_dst = self.g.edges()
            d_src = all_degrees[E_src]
            d_dst = all_degrees[E_dst]
            edge_weights = 1 / (d_src * d_dst).sqrt()
            
            self.g.edata['ew'] = edge_weights.to(self.device)
            self.fn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
            self.fn_reduce = dgl.function.sum(msg='m', out='h')
            
        self.optimizers = {}
        if not self.config['freeze_emb']:
            if self.config['use_sparse']:
                self.optimizers['emb_table-SparseAdam'] = torch.optim.SparseAdam(
                    [{'params':list(self.emb_table.parameters()), 
                      'lr': self.config['emb_lr']}]
                )
            else:
                self.optimizers['emb_table-Adam'] = torch.optim.Adam(
                    [{'params': self.emb_table.parameters(),
                      'lr': self.config['emb_lr']}]
                )
                    
    def _get_user_output_emb(self, users=None):
        if users is None:
            with self.g.local_scope():
                self.g.srcdata['h'] = self.emb_table.weight
                self.g.update_all(self.fn_msg, self.fn_reduce)
                if self.graph_type == 'user-item':
                    aggregated_item_emb = self.g.dstdata['h'][:self.num_users]
                else:
                    aggregated_item_emb = self.g.dstdata['h']
            
            if self.graph_type == 'user-item':
                user_self_emb = self.emb_table.weight[:self.num_users]
            else:
                user_self_emb = self.emb_table.weight
        else:
            input_items, _, blocks = self.node_collator.collate(users.to(self.device))
            block = blocks[0]
            
            with block.local_scope():
                block.srcdata['h'] = self.emb_table(input_items.to(self.device))
                block.update_all(self.fn_msg, self.fn_reduce)
                aggregated_item_emb = block.dstdata['h']
            
            user_self_emb = self.emb_table(users.to(self.device))
            
        theta = self.config['theta']
        user_output_emb = theta * user_self_emb + (1 - theta) * aggregated_item_emb
        
        return user_output_emb
        
    def forward_and_backward(self, batch_data):
        ((src, pos, neg), ) = batch_data
        
        src_emb = self._get_user_output_emb(src.to(self.device))
        pos_emb = self.emb_table(pos.to(self.device))
        neg_emb = self.emb_table(neg.to(self.device))
        
        loss = cosine_contrastive_loss(src_emb, pos_emb, neg_emb, 
                                       self.config['margin'], self.config['neg_weight'])
        
        if self.config['L2_reg_weight'] > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * ((src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum())
            loss += self.config['L2_reg_weight'] * L2_reg_loss
        
        self._backward(loss)
        
        return loss.item()
    
    @torch.no_grad()
    def infer_out_emb_table(self):
        if self.graph_type == 'user-item':
            self.out_emb_table[:self.num_users] = self._get_user_output_emb().cpu()
            self.out_emb_table[self.num_users:] = self.emb_table.weight[self.num_users:].cpu()
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.out_emb_table = self.emb_table.weight
            self.target_emb_table = self.out_emb_table

    def save(self, root=None):
        self._save_optimizers(root)
        self._save_emb_table(root)
        self._save_out_emb_table(root)
    
    def load(self, root=None):
        self._load_optimizers(root)
        self._load_emb_table(root)
        self._load_out_emb_table(root)
