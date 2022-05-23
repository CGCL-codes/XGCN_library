from model.BaseGNNModel import BaseGNNModel

import torch
import dgl


class GAT_Module(torch.nn.Module):
    
    def __init__(
            self, 
            arch="[{'in_feats': 64, 'out_feats': 64, 'num_heads': 4, 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'num_heads': 4}]"
        ):
        super().__init__()
        if isinstance(arch, str):
            arch = eval(arch)
        self.gnn_layers = torch.nn.ModuleList([
            dgl.nn.GATConv(**layer_arch) for layer_arch in arch
        ])
    
    def forward(self, blocks, x):
        assert len(blocks) == len(self.gnn_layers)
        for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
            x = gnn_layer(block, x)
            x = x.mean(dim=-2)  # merge the outputs of different heads
        return x


class GAT(BaseGNNModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        self.gnn = GAT_Module(config['gnn_arch']).to(self.device)
        self.param_list.append({'params': self.gnn.parameters(), 'lr': config['gnn_lr']})
