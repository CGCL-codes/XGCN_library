from XGCN.model.base import BaseEmbeddingModel
from XGCN.model.module.propagation import LightGCN_Propagation
from XGCN.model.module import init_emb_table, dot_product, bpr_loss, bce_loss
from .module import RefNet

import torch
import os.path as osp


class xGCN(BaseEmbeddingModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.emb_table_device = self.config['emb_table_device']
        self.forward_device = self.config['forward_device']
        self.out_emb_table_device = self.config['out_emb_table_device']
        
        self.propagation = LightGCN_Propagation(self.config, self.data)
        
        assert self.config['freeze_emb']
        self.emb_table = init_emb_table(self.config, 
                                        self.info['num_nodes'],
                                        return_tensor=True)  # on self.emb_table_device
        self.create_refnet()  # on self.forward_device
        
        self.out_emb_table = torch.empty(
            self.emb_table.shape, dtype=torch.float32).to(self.out_emb_table_device)
        
        self.node_dl = torch.utils.data.DataLoader(
            torch.arange(self.info['num_nodes']), batch_size=4096)
        
        self.optimizers = {}
        self.optimizers['dnn-Adam'] = torch.optim.Adam([
            {'params': self.refnet.parameters(), 'lr': self.config['dnn_lr']},
        ])
        
        self.epoch_last_prop = 0
        self.total_prop_times = 0
        
        if not self.config['from_pretrained']:
            self.do_propagation()
    
    def forward_and_backward(self, batch_data):
        loss = self.forward(batch_data)
        optmizer = self.optimizers['dnn-Adam']
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        return loss.item()
        
    def create_refnet(self):
        dnn_arch = self.config['dnn_arch']
        if self.config['use_scale_net']:
            scale_net_arch = self.config['scale_net_arch']
        else:
            scale_net_arch = None
        self.refnet = RefNet(dnn_arch, scale_net_arch).to(self.forward_device)
    
    def forward(self, batch_data):
        ((src, pos, neg), ) = batch_data
        
        src_emb = self.get_refnet_output_emb(src)
        pos_emb = self.get_refnet_output_emb(pos)
        neg_emb = self.get_refnet_output_emb(neg)
        
        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)
        
        loss_type = self.config['loss_type']
        if loss_type == 'bpr':
            loss = bpr_loss(pos_score, neg_score)
        elif loss_type == 'bce':
            loss = bce_loss(pos_score, neg_score)
        else:
            assert 0
        
        rw = self.config['L2_reg_weight']
        if rw > 0:
            L2_reg_loss = 1/2 * (1 / len(src)) * (
                (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
            )
            loss += rw * L2_reg_loss
        
        return loss
    
    def get_refnet_output_emb(self, nids):
        emb = self.refnet(self.emb_table[nids].to(self.forward_device))
        return emb
    
    @torch.no_grad()
    def do_propagation(self):
        print("# propagation...")
        self.emb_table = self.propagation(self.emb_table).to(self.emb_table_device)
    
    @torch.no_grad()
    def refresh(self):
        if self.config['renew_by_loading_best'] and (self.total_prop_times >= self.config['K']):
            print("# refresh by loading best...")
            self.emb_table = torch.load(
                osp.join(self.model_root, 'out_emb_table.pt'),
                map_location=self.emb_table_device
            )
        else:
            print("# refresh...")
            self.refnet.eval()
            for nids in self.node_dl:
                self.emb_table[nids] = self.get_refnet_output_emb(nids).to(self.emb_table_device)
            self.refnet.train()
    
    @torch.no_grad()
    def infer_out_emb_table(self):
        self.refnet.eval()
        for nids in self.node_dl:
            self.out_emb_table[nids] = self.get_refnet_output_emb(nids).to(self.out_emb_table_device)

        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table
    
    @torch.no_grad()
    def on_epoch_begin(self):
        epoch = self.data['epoch']
        if self.total_prop_times < self.config['K']:
            if not (epoch % self.config['T']) and epoch != 0:
                self.refresh()
                self.do_propagation()
                self.total_prop_times += 1
                self.epoch_last_prop = epoch
        else:
            if not self.config['use_validation_for_early_stop']:
                # do nothing
                pass
            else:
                if (epoch - self.data['epoch_best']) > self.config['tolerance'] and \
                (epoch - self.epoch_last_prop) > self.config['tolerance']:
                    self.refresh()
                    self.do_propagation()
                    self.total_prop_times += 1
                    self.epoch_last_prop = epoch
        self.refnet.train()
    
    def _save_emb_table(self, root=None):
        if root is None:
            root = self.model_root
        torch.save(self.emb_table, osp.join(root, 'emb_table.pt'))
    
    def _load_emb_table(self, root=None):
        if root is None:
            root = self.model_root
        self.emb_table = torch.load(osp.join(root, 'emb_table.pt'))
    
    def _save_refnet(self, root=None):
        if root is None:
            root = self.model_root
        torch.save(self.refnet.state_dict(), osp.join(root, 'refnet-state_dict.pt'))

    def _load_refnet(self, root=None):
        if root is None:
            root = self.model_root
        state_dict = torch.load(osp.join(root, 'refnet-state_dict.pt'))
        self.refnet.load_state_dict(state_dict)
    
    def save(self, root=None):
        self._save_out_emb_table(root)
        self._save_emb_table(root)
        self._save_refnet(root)
        self._save_optimizers(root)
    
    def load(self, root=None):
        self._load_out_emb_table(root)
        self._load_emb_table(root)
        self._load_refnet(root)
        self._load_optimizers(root)
