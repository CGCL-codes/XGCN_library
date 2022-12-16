from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import dot_product, bpr_loss
from utils import io

import torch
import torch.nn.functional as F
import dgl
import os.path as osp
from tqdm import tqdm


class SSGC(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.device = self.config['device']

        assert self.config['from_pretrained'] and self.config['freeze_emb']
        self.base_emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = torch.empty(self.base_emb_table.weight.shape, dtype=torch.float32)
        
        data_root = self.config['data_root']
        E_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
        E_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
        indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
        
        all_degrees = indptr[1:] - indptr[:-1]
        d_src = all_degrees[E_src]
        d_dst = all_degrees[E_dst]
        
        edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt().to(self.device)
        del indptr, all_degrees, d_src, d_dst
        
        g = dgl.graph((E_src, E_dst)).to(self.device)
        g.edata['ew'] = edge_weights
        g.ndata['X_0'] = self.base_emb_table.weight
        
        transform = dgl.SIGNDiffusion(
            k=self.config['num_gcn_layers'],
            in_feat_name='X_0',
            out_feat_name='X',
            eweight_name='ew'
        )
        g = transform(g)
        
        emb_list = []
        for i in range(1 + self.config['num_gcn_layers']):
            emb_list.append(
                g.ndata['X_' + str(i)]
            )
        self.base_emb_table = torch.stack(emb_list, dim=1).mean(dim=1)
        
        emb_dim = self.base_emb_table.shape[-1]
        self.mlp = torch.nn.Linear(emb_dim, emb_dim).to(self.device)
        
        self.param_list = {
            'Adam': [{'params': self.mlp.parameters(), 'lr': self.config['dnn_lr']}],
        }
    
    def get_output_emb(self, nids):
        return self.mlp(self.base_emb_table[nids])
    
    def __call__(self, batch_data):
        return self.forward(batch_data)
        
    def forward(self, batch_data):
        src, pos, neg = batch_data
        
        src_emb = self.get_output_emb(src)
        pos_emb = self.get_output_emb(pos)
        neg_emb = self.get_output_emb(neg)
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss_fn_type = self.config['loss_fn']
        if loss_fn_type == 'bpr_loss':
            
            loss = bpr_loss(pos_score, neg_score)
        
        elif loss_fn_type == 'bce_loss':
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, 
                torch.ones(pos_score.shape).to(self.device),
            ).mean()
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, 
                torch.zeros(neg_score.shape).to(self.device),
            ).mean()
            
            loss = pos_loss + neg_loss
            
        rw = self.config['l2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * (
                (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
            )
            loss += rw * L2_reg_loss
        
        return loss
    
    def prepare_for_train(self):
        self.mlp.train()
    
    def prepare_for_eval(self):
        self.mlp.eval()
        dl = torch.utils.data.DataLoader(dataset=torch.arange(self.info['num_nodes']), 
                                         batch_size=8192)
        for nids in tqdm(dl, desc="infer all output embs"):
            self.out_emb_table[nids] = self.get_output_emb(nids).cpu()
        self.target_emb_table = self.out_emb_table
        
    def save(self, root, file_out_emb_table=None):
        if file_out_emb_table is None:
            file_out_emb_table = "out_emb_table.pt"
        torch.save(self.out_emb_table, osp.join(root, file_out_emb_table))
