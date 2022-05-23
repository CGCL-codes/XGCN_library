from model.BaseGNNModel import BaseGNNModel

import torch
import dgl


class LightGCN_Module(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
    
    def forward(self, blocks, x):
        # do not stack layers
        for i in range(len(blocks)):
            blocks[i].srcdata['h'] = x
            blocks[i].update_all(self.gcn_msg, self.gcn_reduce)
            x = blocks[i].dstdata['h']
        return x


class Block_LightGCN(BaseGNNModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)        
        self.gnn = LightGCN_Module()
        
        # add edge_weights to the graph
        g = data['node_collate_graph']  # undirected
        src, dst = g.edges()
        degrees = g.out_degrees()
        d1 = degrees[src]
        d2 = degrees[dst]
        edge_weights = (1 / (d1 * d2)).sqrt()
        g.edata['ew'] = edge_weights
