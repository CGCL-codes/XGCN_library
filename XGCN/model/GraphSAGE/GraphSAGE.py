from XGCN.model.base import BaseGNN

import torch
import dgl


class GraphSAGE_Module(torch.nn.Module):
    
    def __init__(
            self, 
            arch=[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]
        ):
        super().__init__()
        if isinstance(arch, str):
            arch = eval(arch)
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.SAGEConv(**layer_arch) for layer_arch in arch
        ])

    def forward(self, graph_or_blocks, x):
        if isinstance(graph_or_blocks, list):
            blocks = graph_or_blocks
            assert len(blocks) == len(self.gnn_layers)
            for block, gnn_layer in zip(blocks, self.gnn_layers):
                x = gnn_layer(block, x)
        else:
            g = graph_or_blocks
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(g, x)
        return x


class GraphSAGE(BaseGNN):
    
    def _create_gnn(self):
        self.gnn = GraphSAGE_Module(arch=self.config['gnn_arch']).to(self.config['gnn_device'])
        self.optimizers['gnn-Adam'] = torch.optim.Adam(
            [{'params': self.gnn.parameters(),
              'lr': self.config['gnn_lr']}]
        )
        
