from XGCN.model.base import BaseEmbeddingModel
from XGCN.model.module import init_emb_table, dot_product, bpr_loss
from XGCN.data import io, csr

import numpy as np
import torch
import torch.nn.functional as F
import dgl
import os.path as osp
from tqdm import tqdm


class MLP(torch.nn.Module):
    '''
        copy from https://github.com/twitter-research/sign/blob/master/sign_training.py 
        and make some changes
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=0.0, activation='F.tanh'):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.activation = eval(activation)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            # x = F.relu(x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        # return torch.log_softmax(x, dim=-1)
        return x


class SIGN(BaseEmbeddingModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.device = self.config['device']

        assert self.config['from_pretrained'] and self.config['freeze_emb']
        self.emb_table = init_emb_table(config, self.info['num_nodes'])
        self.out_emb_table = torch.empty(self.emb_table.weight.shape, dtype=torch.float32)
        
        data_root = self.config['data_root']
        print("# load graph")
        if 'indptr' in self.data:
            indptr = self.data['indptr']
            indices = self.data['indices']
        else:
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            self.data['indptr'] = indptr
            self.data['indices'] = indices
        indptr, indices = csr.get_undirected(indptr, indices)
        E_src = csr.get_src_indices(indptr)
        E_dst = indices
        
        print("# calc edge_weights")
        all_degrees = indptr[1:] - indptr[:-1]
        d_src = all_degrees[E_src]
        d_dst = all_degrees[E_dst]
        
        edge_weights = np.sqrt((1 / (d_src * d_dst)))
        del all_degrees, d_src, d_dst
        
        g = dgl.graph((E_src, E_dst)).to(self.device)
        g.edata['ew'] = torch.FloatTensor(edge_weights).to(self.device)
        g.ndata['X_0'] = self.emb_table.weight
        
        transform = dgl.SIGNDiffusion(
            k=self.config['num_gcn_layers'],
            in_feat_name='X_0',
            out_feat_name='X',
            eweight_name='ew'
        )
        print("# SIGN diffusion...")
        g = transform(g)
        
        emb_list = []
        for i in range(1 + self.config['num_gcn_layers']):
            emb_list.append(
                g.ndata['X_' + str(i)]
            )
        self.emb_table = torch.cat(emb_list, dim=1)
        
        self.mlp = MLP(
            in_channels=self.emb_table.shape[-1],
            hidden_channels=64,
            out_channels=64,
            num_layers=self.config['num_dnn_layers'],
            dropout=0.0,
            activation='torch.tanh'
        ).to(self.device)
        
        self.opt = torch.optim.Adam([
            {'params': self.mlp.parameters(), 'lr': self.config['dnn_lr']}
        ])
    
    def get_output_emb(self, nids):
        return self.mlp(self.emb_table[nids])
    
    def forward_and_backward(self, batch_data):
        ((src, pos, neg), ) = batch_data
        
        src_emb = self.get_output_emb(src)
        pos_emb = self.get_output_emb(pos)
        neg_emb = self.get_output_emb(neg)
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss_fn_type = self.config['loss_fn']
        if loss_fn_type == 'bpr':
            loss = bpr_loss(pos_score, neg_score)
        elif loss_fn_type == 'bce':
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score, 
                torch.ones(pos_score.shape).to(self.device),
            ).mean()
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score, 
                torch.zeros(neg_score.shape).to(self.device),
            ).mean()
            
            loss = pos_loss + neg_loss
        else:
            assert 0
        
        rw = self.config['L2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * (
                (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
            )
            loss += rw * L2_reg_loss
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        return loss.item()
    
    def on_epoch_begin(self):
        self.mlp.train()
    
    @torch.no_grad()
    def on_eval_begin(self):
        self.mlp.eval()
        dl = torch.utils.data.DataLoader(dataset=torch.arange(self.info['num_nodes']), 
                                         batch_size=8192)
        for nids in tqdm(dl, desc="infer all output embs"):
            self.out_emb_table[nids] = self.get_output_emb(nids).cpu()

        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table
