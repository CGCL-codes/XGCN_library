from model.BaseGNNModel import BaseGNNModel

import torch
import dgl


class _MLP(torch.nn.Module):
    
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
        self.gin_mlp_list = torch.nn.ModuleList([_MLP() for _ in range(num_gcn_layers)])
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.GINConv(apply_func=self.gin_mlp_list[i], aggregator_type="sum") 
            for i in range(num_gcn_layers)
        ])
    
    def forward(self, blocks, x):
        assert len(blocks) == len(self.gnn_layers)
        for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
            x = gnn_layer(block, x)
        return x


class GIN(BaseGNNModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.gnn = GIN_Module(config['num_gcn_layers']).to(self.device)
        self.param_list.append({'params': self.gnn.parameters(), 'lr': config['gnn_lr']})
