from ..base import BaseGNN

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
    
    def forward(self, graph_or_blocks, x):
        if isinstance(graph_or_blocks, list):
            blocks = graph_or_blocks
            assert len(blocks) == len(self.gnn_layers)
            for i, (block, gnn_layer) in enumerate(zip(blocks, self.gnn_layers)):
                x = gnn_layer(block, x)
                x = x.mean(dim=-2)  # merge the outputs of different heads
        else:
            g = graph_or_blocks
            for gnn_layer in self.gnn_layers:
                x = gnn_layer(g, x)
                x = x.mean(dim=-2)  # merge the outputs of different heads
        return x

    def get_layer_output(self, block, x, layer_i):
        x = self.gnn_layers[layer_i](block, x)
        x = x.mean(dim=-2)
        return x


class GAT(BaseGNN):
    
    def _create_gnn(self):
        self.gnn = GAT_Module(arch=self.config['gnn_arch']).to(self.config['gnn_device'])
        self.optimizers['gnn-Adam'] = torch.optim.Adam(
            [{'params': self.gnn.parameters(),
              'lr': self.config['gnn_lr']}]
        )
