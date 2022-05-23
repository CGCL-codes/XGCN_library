from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import dot_product, bpr_loss, bce_loss, cosine_contrastive_loss

import torch
import dgl
from tqdm import tqdm
import os.path as osp


class _GCN_Module(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gcn_msg = dgl.function.copy_u('h', 'm')
        self.gcn_reduce = dgl.function.mean(msg='m', out='h')
    
    def forward(self, blocks, x):
        for i in range(len(blocks)):
            blocks[i].srcdata['h'] = x
            blocks[i].update_all(self.gcn_msg, self.gcn_reduce)
            x = blocks[i].dstdata['h']
        return x


class Block_SimpleX(BaseEmbeddingModel):

    def __init__(self, config, data):
        super().__init__(config, data)
        self.device = self.config['device']
        
        self.dataset_type = self.info['dataset_type']
        assert self.dataset_type == 'social'

        self.num_nodes = self.info['num_nodes']
        self.emb_table = init_emb_table(config, self.num_nodes)
        self.gnn = _GCN_Module()

        self.param_list = []
        if not ('freeze_emb' in config and config['freeze_emb']):
            self.param_list.append({'params': self.emb_table, 'lr': config['emb_lr']})
    
    def __call__(self, batch_data):
        return self.forward(batch_data)

    def forward(self, batch_data):
        batch_nids, local_idx, input_nids, output_nids, blocks = batch_data

        blocks = [block.to(self.device) for block in blocks]
        
        output_embs = self.gnn(
            blocks, self.emb_table[input_nids]
        )
        
        input_nids = input_nids[local_idx].view(3, -1)
        src = input_nids[0]
        src_self_emb = self.emb_table[src]
        pos = input_nids[1]
        pos_emb = self.emb_table[pos]
        neg_emb = self.emb_table[
            torch.randint(0, self.num_nodes, (len(pos), 128))
        ]
        
        output_embs = output_embs[local_idx].view(3, -1, self.emb_table.shape[-1])
        src_aggr_emb = output_embs[0, :, :]

        theta = self.config['theta']
        src_emb = theta * src_self_emb + (1 - theta) * src_aggr_emb

        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss = cosine_contrastive_loss(src_emb, pos_emb, neg_emb, 
                                       self.config['margin'], self.config['neg_weight'])

        if self.config['l2_reg_weight'] > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * ((src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum())
            loss += self.config['l2_reg_weight'] * L2_reg_loss
        
        return loss
    
    def prepare_for_train(self):
        pass
    
    def prepare_for_eval(self):
        self.out_emb_table = self.emb_table
        self.target_emb_table = self.out_emb_table
    
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))
