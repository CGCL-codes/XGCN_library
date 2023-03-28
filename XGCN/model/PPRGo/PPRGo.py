from XGCN.model.base import BaseEmbeddingModel
from XGCN.model.module import init_emb_table, dot_product, bpr_loss, bce_loss
from XGCN.data import io

import torch
import os.path as osp
from tqdm import tqdm


class PPRGo(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.ppr_data_device = self.config['ppr_data_device']
        self.emb_table_device = self.config['emb_table_device']
        self.forward_device = self.config['forward_device']
        self.out_emb_table_device = self.config['out_emb_table_device']
        
        self.emb_table = init_emb_table(self.config, self.info['num_nodes'])
        
        print("## load ppr neighbors and ppr weights ...")
        raw_nei = io.load_pickle(osp.join(self.config['ppr_data_root'], "nei.pkl"))
        raw_wei = io.load_pickle(osp.join(self.config['ppr_data_root'], "wei.pkl"))
        
        topk = self.config['topk']
        self.nei = torch.LongTensor(raw_nei[:, : topk])
        self.wei = torch.FloatTensor(raw_wei[:, : topk])
        
        if self.config['use_uniform_weight']:
            print("## uniform weight")
            _w = torch.ones(self.nei.shape)
            _w[self.wei == 0] = 0
            self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            print("## not uniform weight")
            self.wei = self.wei / (self.wei.sum(dim=-1, keepdim=True) + 1e-12)
        
        self.opt_list = []
        if not self.config['freeze_emb']:
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
        
    def calc_pprgo_output_emb(self, nids):
        top_nids = self.nei[nids]
        top_weights = self.wei[nids].to(self.forward_device)
        
        top_embs = self.emb_table(top_nids.to(self.emb_table_device)).to(self.forward_device)
        
        out_embs = torch.matmul(top_weights.unsqueeze(-2), top_embs)
          
        return out_embs.squeeze()
        
    def forward_and_backward(self, batch_data):
        ((src, pos, neg), ) = batch_data
        
        src_emb = self.calc_pprgo_output_emb(src)
        pos_emb = self.calc_pprgo_output_emb(pos)
        neg_emb = self.calc_pprgo_output_emb(neg)
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss_type = self.config['loss_type']
        if loss_type == 'bpr':
            loss = bpr_loss(pos_score, neg_score)
        elif loss_type == 'bce':
            loss = bce_loss(pos_score, neg_score)
        else:
            assert 0
        
        rw = self.config['L2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * (
                (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
            )
            loss += rw * L2_reg_loss
        
        self.backward(loss)
        return loss.item()
    
    @torch.no_grad()
    def on_eval_begin(self):
        self.out_emb_table = torch.empty(size=self.emb_table.weight.shape, dtype=torch.float32,
                                         device=self.out_emb_table_device)
        dl = torch.utils.data.DataLoader(dataset=torch.arange(self.info['num_nodes']), 
                                         batch_size=8192)
        for nids in tqdm(dl, desc="infer pprgo output emb"):
            self.out_emb_table[nids] = self.calc_pprgo_output_emb(nids).to(self.out_emb_table_device)
        self.target_emb_table = self.out_emb_table
        
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table

    def backward(self, loss):
        for opt in self.opt_list:
            opt.zero_grad()
        loss.backward()
        for opt in self.opt_list:
            opt.step()
