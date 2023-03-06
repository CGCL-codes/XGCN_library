from .model import JK_GAMLP, R_GAMLP

from XGCN.model.base import BaseGNN

import torch
import dgl


class GAMLP_learnable_emb_Module(torch.nn.Module):
    
    def __init__(self, GAMLPl_type, emb_dim, num_gcn_layers):
        super().__init__()
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
        
        if GAMLPl_type == 'GAMLP_JK':
            MLP = JK_GAMLP
        elif GAMLPl_type == 'GAMLP_R':
            MLP = R_GAMLP
        else:
            assert 0
        
        self.mlp = MLP(
            nfeat=emb_dim,
            hidden=512,
            nclass=emb_dim,  # output node emb, rather than logits for classification
            num_hops=num_gcn_layers + 1,
            dropout=0.0,
            input_drop=0.0,
            att_dropout=0.0,
            alpha=0.5,
            n_layers_1=4,
            n_layers_2=4,
            act='torch.tanh',
            pre_process=False,
            residual=False,
            pre_dropout=False,
            bns=False
        )
        
    def forward(self, blocks, x):
        num_output_nodes = blocks[-1].number_of_dst_nodes()
        x_of_different_layers = [
            x[:num_output_nodes], # embeddings of layer zero
        ]
        for block in blocks:
            block.srcdata['h'] = x
            block.update_all(self.gcn_msg, self.gcn_reduce)
            x = block.dstdata['h']
            x_of_different_layers.append(x[:num_output_nodes])
        return self.mlp(x_of_different_layers)


class GAMLP_learnable_emb(BaseGNN):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        
        # add edge_weights to the graph
        g = data['node_collate_graph']  # undirected
        src, dst = g.edges()
        degrees = g.out_degrees()
        d1 = degrees[src]
        d2 = degrees[dst]
        edge_weights = (1 / (d1 * d2)).sqrt()
        g.edata['ew'] = edge_weights
        
        self.gnn = GAMLP_learnable_emb_Module(
            GAMLPl_type=self.config['GAMLP_type'],
            emb_dim=self.config['emb_dim'],
            num_gcn_layers=self.config['num_gcn_layers']
        ).to(self.device)
        self.param_list.append({'params': self.gnn.parameters(), 'lr': config['gnn_lr']})
