from utils import io
from utils.metric import all_metrics, multi_pos_all_metrics#, new_multi_pos_all_metrics
from model.module import dot_product
from data.csr_graph_helper import neighbors

import numpy as np
import numba
import torch
import os.path as osp


def init_emb_table(config, num_nodes=None):
    if 'from_pretrained' in config and config['from_pretrained']:
        file_pretrained_emb = config['file_pretrained_emb']
        print('## load pretrained embedding:', file_pretrained_emb)
        emb_table = torch.load(file_pretrained_emb, map_location=config['device'])
    else:
        emb_table = torch.FloatTensor(size=(num_nodes, config['emb_dim']))
        torch.nn.init.normal_(emb_table, mean=0.0, std=config['emb_init_std'])
    
    emb_table = emb_table.to(config['device'])
    
    if not ('freeze_emb' in config and config['freeze_emb']):
        emb_table.requires_grad = True
    else:
        emb_table.requires_grad = False
    
    return emb_table


@numba.jit(nopython=True)
def mask_neighbor_score(csr_graph_indptr, csr_graph_indices,
                        src, all_item_score, num_user):
    for i, u in enumerate(src):
        nei_item_id = neighbors(csr_graph_indptr, csr_graph_indices, u) - num_user
        all_item_score[i][nei_item_id] = -999999


@numba.jit(nopython=True)
def _mask_neighbor_score(csr_graph_indptr, csr_graph_indices,
                         src, all_target_score):
    for i, u in enumerate(src):
        nei_target_id = neighbors(csr_graph_indptr, csr_graph_indices, u)
        all_target_score[i][nei_target_id] = -999999


class BaseEmbeddingModel:
    
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.info = io.load_yaml(osp.join(self.config['data_root'], 'info.yaml'))
    
        self.param_list = None
        
        self.dataset_type = self.info['dataset_type']
        if self.dataset_type == 'user-item':
            self.num_users = self.info['num_users']
        
        # for evaluation:
        self.out_emb_table = None
        self.target_emb_table = None
        
        self.one_pos = False
        self.eval_on_whole_graph = False
        self.mask_nei = False
        
        self.train_csr_indptr = None
        self.train_csr_indices = None
    
    def parameters(self):
        return self.param_list
    
    def prepare_csr_graph(self):
        if 'train_csr_indptr' in self.data:
            self.train_csr_indptr = self.data['train_csr_indptr']
            self.train_csr_indices = self.data['train_csr_indices']
        else:
            data_root = self.config['data_root']
            self.train_csr_indptr = io.load_pickle(osp.join(data_root, 'train_csr_indptr.pkl'))
            self.train_csr_indices = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    
    def load(self, root):
        self.out_emb_table = torch.load(osp.join(root, "out_emb_table.pt"))
        if self.dataset_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.num_users:]
        else:  # 'social'
            self.target_emb_table = self.out_emb_table
        
    def eval_a_batch(self, batch_data):
        if self.eval_on_whole_graph:
            src, pos = batch_data
            num_batch_samples = len(src)
            
            src_emb = self.out_emb_table[src]
            all_target_score = (src_emb @ self.target_emb_table.t()).cpu().numpy()
            
            if self.mask_nei:
                if self.train_csr_indptr is None:
                    self.prepare_csr_graph()
                _mask_neighbor_score(self.train_csr_indptr, self.train_csr_indices,
                                     src, all_target_score)
            
            if self.one_pos:
                pos_emb = self.target_emb_table[pos]
                pos_score = dot_product(src_emb, pos_emb)
                pos_neg_score = np.concatenate((pos_score.view(-1, 1).cpu().numpy(), all_target_score), axis=-1)
                batch_results = all_metrics(pos_neg_score)
            else:
                pos_list = pos
                assert isinstance(pos_list, list)
                batch_results = multi_pos_all_metrics(pos, all_target_score)
                
        else:
            assert self.one_pos == True and self.mask_nei == False
            src, pos, neg = batch_data
            num_batch_samples = len(src)
            
            src_emb = self.out_emb_table[src]
            pos_emb = self.target_emb_table[pos]
            neg_emb = self.target_emb_table[neg]
            
            pos_score = dot_product(src_emb, pos_emb)
            neg_score = dot_product(src_emb, neg_emb)
            
            pos_neg_score = torch.cat((pos_score.view(-1, 1), neg_score), dim=-1).cpu().numpy()
            batch_results = all_metrics(pos_neg_score)
        
        return batch_results, num_batch_samples
