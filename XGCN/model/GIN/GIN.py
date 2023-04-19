from ..base import BaseGNN

import torch
import dgl


class MLP(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Sequential(
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
        )
    
    def forward(self, x):
        return self.m(x)


class GIN_Module(torch.nn.Module):
    
    def __init__(self, num_gcn_layers):
        super().__init__()
        self.gin_mlp_list = torch.nn.ModuleList([MLP() for _ in range(num_gcn_layers)])
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.GINConv(apply_func=self.gin_mlp_list[i], aggregator_type="sum") 
            for i in range(num_gcn_layers)
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


class GIN(BaseGNN):
    
    def _create_gnn(self):
        self.gnn = GIN_Module(num_gcn_layers=self.config['num_gcn_layers']).to(self.config['gnn_device'])
        self.optimizers['gnn-Adam'] = torch.optim.Adam(
            [{'params': self.gnn.parameters(),
              'lr': self.config['gnn_lr']}]
        )
