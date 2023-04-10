from XGCN.model.base import BaseGNN
from .SIGN import MLP

import torch
import dgl
from tqdm import tqdm


class SIGN_learnable_emb_Module(torch.nn.Module):
    
    def __init__(self, emb_dim, num_gcn_layers, num_dnn_layers):
        super().__init__()
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
        self.mlp = MLP(
            in_channels=emb_dim * (num_gcn_layers + 1),
            hidden_channels=emb_dim,
            out_channels=emb_dim,
            num_layers=num_dnn_layers,
            dropout=0.0,
            activation='torch.tanh'
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
        xs_cat = torch.cat(x_of_different_layers, dim=-1)
        return self.mlp(xs_cat)

    
class SIGN_learnable_emb(BaseGNN):
    
    def _create_gnn(self):
        src, dst = self.g.edges()
        degrees = self.g.out_degrees()
        d1 = degrees[src]
        d2 = degrees[dst]
        edge_weights = (1 / (d1 * d2)).sqrt()
        self.g.edata['ew'] = edge_weights
              
        self.gnn = SIGN_learnable_emb_Module(
            emb_dim=self.config['emb_dim'], 
            num_gcn_layers=self.config['num_gcn_layers'],
            num_dnn_layers=self.config['num_dnn_layers']
        ).to(self.config['gnn_device'])
        self.optimizers.append(
            torch.optim.Adam([{'params': self.gnn.parameters(),
                                'lr': self.config['gnn_lr']}])
        )

    @torch.no_grad()
    def on_eval_begin(self):
        block_sampler = self.data['block_sampler']
        self.out_emb_table = torch.empty(self.emb_table.weight.shape, dtype=torch.float32).to(self.config['out_emb_table_device'])
        dl = torch.utils.data.DataLoader(dataset=torch.arange(self.info['num_nodes']), 
                                         batch_size=8192)
        
        for nids in tqdm(dl, desc="get all gnn output embs"):
            input_nids, _, blocks = block_sampler.sample_blocks(self.g, nids.to(self.g.device))
            blocks = [block.to(self.config['gnn_device']) for block in blocks]
            output_embs = self.gnn(
                blocks, self.emb_table(input_nids.to(self.config['emb_table_device']))
            )
            self.out_emb_table[nids] = output_embs
        
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
