from model.BaseGNNModel import BaseGNNModel

import torch
import dgl


class GraphSAGE_Module(torch.nn.Module):
    
    def __init__(
            self, 
            arch="[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]"
        ):
        super().__init__()
        if isinstance(arch, str):
            arch = eval(arch)
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.SAGEConv(**layer_arch) for layer_arch in arch
        ])

    def forward(self, blocks, x):
        assert len(blocks) == len(self.gnn_layers)
        for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
            x = gnn_layer(block, x)
        return x


class GraphSAGE(BaseGNNModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)        
        self.gnn = GraphSAGE_Module(config['gnn_arch']).to(self.device)
        self.param_list.append({'params': self.gnn.parameters(), 'lr': config['gnn_lr']})
