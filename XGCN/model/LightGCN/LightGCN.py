from ..base import BaseGNN

import torch
import dgl


class LightGCNConv(torch.nn.Module):
    
    def __init__(self, num_layers, stack_layers=True):
        super().__init__()
        self.num_layers = num_layers
        self.stack_layers = stack_layers
        self.gcn_msg = dgl.function.u_mul_e('h', 'ew', 'm')
        self.gcn_reduce = dgl.function.sum(msg='m', out='h')
    
    def forward(self, graph_or_blocks, x):
        if isinstance(graph_or_blocks, list):
            blocks = graph_or_blocks
            x_out = self.block_forward(blocks, x)
        else:
            graph = graph_or_blocks
            x_out = self.graph_forward(graph, x)
        return x_out
    
    def block_forward(self, blocks, x):
        if self.stack_layers:
            num_output_nodes = blocks[-1].number_of_dst_nodes()
            x_of_different_layers = [
                x[:num_output_nodes], # embeddings of layer zero
            ]
            for block in blocks:
                block.srcdata['h'] = x
                block.update_all(self.gcn_msg, self.gcn_reduce)
                x = block.dstdata['h']
                x_of_different_layers.append(x[:num_output_nodes])
            # take the mean of each layer:
            x_out = torch.stack(x_of_different_layers, dim=1).mean(dim=1)
        else:
            for block in blocks:
                block.srcdata['h'] = x
                block.update_all(self.gcn_msg, self.gcn_reduce)
                x = block.dstdata['h']
            x_out = x
        return x_out
        
    def graph_forward(self, graph, x):
        if self.stack_layers:
            x_of_different_layers = [x]
            for _ in range(self.num_layers):
                graph.srcdata['h'] = x_of_different_layers[-1]
                graph.update_all(self.gcn_msg, self.gcn_reduce)
                x_of_different_layers.append(graph.dstdata['h'])
            # take the mean of each layer:
            X_out = torch.stack(x_of_different_layers, dim=1).mean(dim=1)
        else:
            for _ in range(self.num_layers):
                graph.srcdata['h'] = x
                graph.update_all(self.gcn_msg, self.gcn_reduce)
                x = graph.dstdata['h']
            X_out = x
        return X_out
        

class LightGCN(BaseGNN):
    
    def __init__(self, config):
        super().__init__(config)
        
    def _create_gnn(self):
        all_degrees = self.g.out_degrees()
        E_src, E_dst = self.g.edges()
        d_src = all_degrees[E_src]
        d_dst = all_degrees[E_dst]
        
        edge_weights = 1 / (d_src * d_dst).sqrt()
        self.g.edata['ew'] = edge_weights.to(self.graph_device)
        self.gnn = LightGCNConv(
            num_layers=self.config['num_gcn_layers'],
            stack_layers=self.config['stack_layers']
        )
    
    @torch.no_grad()
    def infer_out_emb_table(self):
        if self.forward_mode == 'full_graph':
            self.out_emb_table = self.gnn(
                self.g, self.emb_table.weight
            ).to(self.out_emb_table_device)
        else:
            self.out_emb_table = self.gnn(
                self.g.to('cpu'), self.emb_table.weight.to('cpu')
            ).to(self.out_emb_table_device)
            self.emb_table.weight.to(self.emb_table_device)
        
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table
