import XGCN
from XGCN.model.base import BaseModel
from XGCN.model.module.mask_neighbor_score import mask_neighbor_score, mask_neighbor_score_user_item
from XGCN.model.module import bpr_loss, bce_loss
from XGCN.data import io, csr
from XGCN.utils.utils import ensure_dir, print_dict, get_formatted_results
from .hashing import ElphHashes

import torch
from torch import nn
import numpy as np
import os.path as osp


class ELPH(BaseModel):

    def __init__(self, config):
        print_dict(config)
        self.config = config
        self.data = {}
        self._init()
    
    def _init(self):
        self.device = self.config['device']
        self.data_root = self.config['data_root']
        assert osp.exists(self.data_root)
        
        self.results_root = self.config['results_root']
        ensure_dir(self.results_root)
        io.save_yaml(osp.join(self.results_root, 'config.yaml'), self.config)
        
        self.model_root = osp.join(self.results_root, 'model')
        ensure_dir(self.model_root)
        
        self.info = io.load_yaml(osp.join(self.data_root, 'info.yaml'))
        self.graph_type = self.info['graph_type']
        if self.graph_type == 'user-item':
            self.num_users = self.info['num_users']
        
        indptr = io.load_pickle(osp.join(self.data_root, 'indptr.pkl'))
        indices = io.load_pickle(osp.join(self.data_root, 'indices.pkl'))
        
        self.indptr = indptr
        self.indices = indices

        un_indptr, un_indices = csr.get_undirected(indptr, indices)
        src_indices = csr.get_src_indices(un_indptr)
        edge_index = torch.LongTensor(np.stack([src_indices, un_indices]))
    
        self.heuristic = ElphHashes(self.config, self.info['num_nodes'], edge_index)

        if self.info['graph_type'] == 'user-item':
            self.all_target_nodes = self.info['num_users'] + torch.arange(self.info['num_items'])
        else:
            self.all_target_nodes = torch.arange(self.info['num_nodes'])

        self.mlp = torch.nn.Sequential(*eval(self.config['dnn_arch'])).to(self.device)
        self.opt = torch.optim.Adam([
            {'params': self.mlp.parameters(), 'lr': self.config['dnn_lr']},
        ])

    def fit(self):
        config = self.config
        data = self.data
        model = self
        train_dl = XGCN.create_DataLoader(config, data)
        
        self.trainer = XGCN.create_Trainer(
            config, data, model, train_dl
        )
        self.trainer.train()
        
        if self.config['use_validation_for_early_stop']:
            self.load()
        
    def test(self, test_config=None):
        if test_config is None:
            test_config = self.config
        
        test_evaluator = XGCN.create_test_Evaluator(
            config=test_config, data=self.data, model=self
        )
        results = test_evaluator.eval(desc='test')
        
        results['formatted'] = get_formatted_results(results)
        return results

    def forward_and_backward(self, batch_data):
        src, pos, neg = batch_data
        
        pos_score = self.predict(src, pos)
        neg_score = self.predict(src, neg)
        
        loss_type = self.config['loss_type']
        if loss_type == 'bpr':
            loss = bpr_loss(pos_score, neg_score)
        elif loss_type == 'bce':
            loss = bce_loss(pos_score, neg_score)
        else:
            assert 0
        
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.item()
    
    def get_input_dim(self):
        a = self.config['max_hash_hops']
        return a * (a + 2)
    
    def get_link_heuristic(self, links):
        x = self.heuristic.get_link_heuristic(links)
        return x
    
    def predict(self, u, v):
        links = torch.cat([u.reshape(-1, 1), v.reshape(-1, 1)], dim=-1)
        x = self.get_link_heuristic(links)
        logits = self.mlp(x)
        return logits

    def get_edge_emb(self, u, v):
        heu_feat = self.heu.get_feat(u.numpy(), v.numpy())
        return heu_feat

    @torch.no_grad()
    def infer_all_target_score(self, src, mask_nei=True):
        src = torch.LongTensor(src)
        target = torch.Tensor.repeat(self.all_target_nodes, len(src))
        src = torch.repeat_interleave(src, len(self.all_target_nodes))
        links = torch.cat([src.reshape(-1, 1), target.reshape(-1, 1)], dim=-1)
        
        x = self.get_link_heuristic(links)
        scores = self.mlp(x)
        scores = scores.reshape(-1, len(self.all_target_nodes))
        all_target_score = scores.cpu().numpy()
        
        if mask_nei:
            self.mask_neighbor_score(src.numpy(), all_target_score)
        
        return all_target_score
        
    def mask_neighbor_score(self, src, all_target_score):
        if self.graph_type == 'user-item':
            mask_neighbor_score_user_item(self.indptr, self.indices,
                src, all_target_score, self.num_users
            )
        else:
            mask_neighbor_score(self.indptr, self.indices,
                src, all_target_score
            )
    
    def _eval_a_batch(self, batch_data, eval_type):
        return {
            'whole_graph_multi_pos': self._eval_whole_graph_multi_pos,
            'whole_graph_one_pos': self._eval_whole_graph_one_pos,
        }[eval_type](batch_data)
    
    @torch.no_grad()
    def _eval_whole_graph_multi_pos(self, batch_data):
        src, _ = batch_data
        
        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        
        return all_target_score
    
    @torch.no_grad()
    def _eval_whole_graph_one_pos(self, batch_data):
        src, pos = batch_data

        all_target_score = self.infer_all_target_score(src, mask_nei=True)
        
        pos_score = np.empty((len(src),), dtype=np.float32)
        for i in range(len(src)):
            pos_score[i] = all_target_score[i][pos[i]]
        pos_neg_score = np.concatenate((pos_score.reshape(-1, 1), all_target_score), axis=-1)
        
        return pos_neg_score
    
    def save(self, root=None):
        if root is None:
            root = self.model_root
        torch.save(self.mlp, osp.join(root, 'mlp.pt'))
        torch.save(self.opt.state_dict(), osp.join(root, 'opt-state_dict.pt'))

    def load(self, root=None):
        if root is None:
            root = self.model_root
        self.mlp = torch.load(osp.join(root, 'mlp.pt'))
        self.opt.load_state_dict(torch.load(osp.join(root, 'opt-state_dict.pt')))
