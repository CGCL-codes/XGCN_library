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


class MyDNN(torch.nn.Module):
    
    def __init__(self, dnn_arch, scale_net_arch):
        super(MyDNN, self).__init__()
        self.dnn = torch.nn.Sequential(*eval(dnn_arch))
        # self.dnn = torch.nn.Sequential(
        #     torch.nn.Linear(64, 1024), 
        #     torch.nn.Tanh(), 
        #     torch.nn.Linear(1024, 1024), 
        #     torch.nn.Tanh(), 
        #     torch.nn.Linear(1024, 64)
        # )
        
        self.scale_net = torch.nn.Sequential(*eval(scale_net_arch))
        # self.scale_net = torch.nn.Sequential(
        #     torch.nn.Linear(64, 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 1),
        #     torch.nn.Sigmoid()
        # )
    
    def forward(self, X):
        
        theta = self.scale_net(X)
        
        X = theta * self.dnn(X)
        
        return X


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
        
        self.use_dnn = ('use_special_dnn' in self.config and self.config['use_special_dnn'])
        if self.use_dnn:
            if len(eval(self.config['scale_net_arch'])) == 0:
                print("# use FFN to transform lightgcn output emb")
                self.dnn = torch.nn.Sequential(*eval(self.config['dnn_arch'])).to(self.device)
            else:
                print("# use FFN + SSNet to transform lightgcn output emb")
                self.dnn = MyDNN(self.config['dnn_arch'], self.config['scale_net_arch']).to(self.device)
            self.param_list.append({'params': self.dnn.parameters(), 'lr': config['emb_lr']})
    
    def forward(self, batch_data):
        batch_nids, local_idx, input_nids, output_nids, blocks = batch_data
    
        blocks = [block.to(self.device) for block in blocks]
        
        output_embs = self.gnn(
            blocks, self.base_emb_table(input_nids.to(self.device))
        )
        
        output_embs = output_embs[local_idx].view(3, -1, self.base_emb_table.weight.shape[-1])
        
        if self.use_dnn:
            output_embs = self.dnn(output_embs)
        
        src_emb = output_embs[0, :, :]
        pos_emb = output_embs[1, :, :]
        neg_emb = output_embs[2, :, :]
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss = bpr_loss(pos_score, neg_score)

        rw = self.config['l2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * (1 / len(output_embs)) * (output_embs**2).sum()
            loss += rw * L2_reg_loss
        
        return loss
      
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
                if self.use_dnn:
                    output_embs = self.dnn(output_embs)
                self.out_emb_table[output_nids] = output_embs
            
        else:
            self.out_emb_table = self._gnn(self.g, self.base_emb_table.cpu().weight)
            self.base_emb_table = self.base_emb_table.to(self.device)
            
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:
            self.target_emb_table = self.out_emb_table
