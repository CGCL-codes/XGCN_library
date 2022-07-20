from model.BaseGNNModel import BaseGNNModel
from model.LightGCN import LightGCNConv

import torch
import dgl
import os.path as osp
from tqdm import tqdm


class Block_LightGCNConv(torch.nn.Module):
    
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
        self.gnn = Block_LightGCNConv()
        
        # add edge_weights to the graph
        g = data['node_collate_graph']  # undirected
        src, dst = g.edges()
        degrees = g.out_degrees()
        d1 = degrees[src]
        d2 = degrees[dst]
        edge_weights = (1 / (d1 * d2)).sqrt()
        g.edata['ew'] = edge_weights
        self.g = g
        self._gnn = LightGCNConv(config['num_gcn_layers'], stack_layers=False)
    
    def save(self, root):
        torch.save(self.out_emb_table, osp.join(root, "out_emb_table.pt"))

    def prepare_for_eval(self):
        if len(eval(self.config['num_layer_sample'])) != 0:  # use neighbor sampling
            node_collator = self.data['node_collator']
            self.out_emb_table = torch.empty(self.base_emb_table.weight.shape, dtype=torch.float32).to(self.device)
            dl = torch.utils.data.DataLoader(dataset=torch.arange(self.info['num_nodes']), 
                                            batch_size=8192,
                                            collate_fn=node_collator.collate)
            
            for input_nids, output_nids, blocks in tqdm(dl, desc="get all gnn output embs"):
                blocks = [block.to(self.device) for block in blocks]
                output_embs = self.gnn(
                    blocks, self.base_emb_table(input_nids.to(self.device))
                )
                self.out_emb_table[output_nids] = output_embs
            
        else:
            self.out_emb_table = self._gnn(self.g, self.base_emb_table.cpu().weight)
            self.base_emb_table = self.base_emb_table.to(self.device)
            
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
