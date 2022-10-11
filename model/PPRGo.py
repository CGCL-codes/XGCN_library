from utils import io
from model.BaseEmbeddingModel import BaseEmbeddingModel, init_emb_table
from model.module import dot_product, bpr_loss, ssm_loss

import torch
import torch.nn.functional as F
import os.path as osp
from tqdm import tqdm


class SSNet(torch.nn.Module):
    
    def __init__(self):
        super(SSNet, self).__init__()
        self.snn = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, X):
        return self.snn(X) * X


class PPRGo(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.device = self.config['device']
        
        self.base_emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = torch.empty(self.base_emb_table.weight.shape, dtype=torch.float32)
        
        assert config['use_sparse']
        
        self.param_list = {
            'SparseAdam': [],
        }
        if not self.config['freeze_emb']:
            self.param_list['SparseAdam'].append({'params': list(self.base_emb_table.parameters()),
                                    'lr': config['emb_lr']})
        
        print("## load ppr neighbors and ppr weights ...")
        raw_nei = io.load_pickle(osp.join(config['ppr_data_root'], "nei.pkl"))
        raw_wei = io.load_pickle(osp.join(config['ppr_data_root'], "wei.pkl"))
        
        topk = config['topk']
        self.nei = torch.LongTensor(raw_nei[:, : topk])
        self.wei = torch.FloatTensor(raw_wei[:, : topk])
        
        if config['use_uniform_weight']:
            print("## uniform weight")
            _w = torch.ones(self.nei.shape)
            _w[self.wei == 0] = 0
            self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)
        else:
            print("## not uniform weight")
            self.wei = self.wei / (self.wei.sum(dim=-1, keepdim=True) + 1e-12)
        
        self.use_dnn = (
            'use_special_dnn' in self.config and self.config['use_special_dnn']
        )
        if self.use_dnn:
            print("## use SSNet to re-scale output emb")
            self.dnn = SSNet().to(self.device)
            self.param_list['Adam'] = [{'params': self.dnn.parameters(), 'lr': 0.001}]
        
    def _calc_pprgo_out_emb(self, nids):
        top_nids = self.nei[nids].to(self.device)
        top_weights = self.wei[nids].to(self.device)
        
        top_embs = self.base_emb_table(top_nids)
        
        out_embs = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        
        if self.use_dnn:
            out_embs = self.dnn(out_embs)
        
        return out_embs.squeeze()
        
    def __call__(self, batch_data):
        return self.forward(batch_data)
        
    def forward(self, batch_data):
        src, pos, neg = batch_data
        
        src_emb = self._calc_pprgo_out_emb(src)
        pos_emb = self._calc_pprgo_out_emb(pos)
        
        loss_fn_type = self.config['loss_fn']
        if loss_fn_type == 'bpr_loss':
            neg_emb = self._calc_pprgo_out_emb(neg)
            
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            
            loss = bpr_loss(pos_score, neg_score)
            
            rw = self.config['l2_reg_weight']
            if rw > 0:
                L2_reg_loss = 1/2 * (1 / len(src)) * (
                    (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
                )
                loss += rw * L2_reg_loss
        
        elif loss_fn_type == 'bce_loss':
            neg_emb = self._calc_pprgo_out_emb(neg)
            
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            
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
        
        elif loss_fn_type == 'ssm_loss':
            
            loss = ssm_loss(src_emb, pos_emb, tao=self.config['tao'])
            
            rw = self.config['l2_reg_weight']
            if rw > 0:
                L2_reg_loss = 1/2 * (1 / len(src)) * (
                    (src_emb**2).sum() + (pos_emb**2).sum()
                )
                loss += rw * L2_reg_loss
        else:
            assert 0
        
        return loss
    
    def prepare_for_train(self):
        pass
    
    def prepare_for_eval(self):
        dl = torch.utils.data.DataLoader(dataset=torch.arange(self.info['num_nodes']), 
                                         batch_size=8192)
        for nids in tqdm(dl, desc="infer pprgo output embs"):
            self.out_emb_table[nids] = self._calc_pprgo_out_emb(nids).cpu()
        self.target_emb_table = self.out_emb_table
        
        torch.save(self.base_emb_table.weight,
                   osp.join(osp.join(self.config['results_root'], 'base_emb_table.pt')))
        
    def save(self, root, file_out_emb_table=None):
        if file_out_emb_table is None:
            file_out_emb_table = "out_emb_table.pt"
        torch.save(self.out_emb_table, osp.join(root, file_out_emb_table))
