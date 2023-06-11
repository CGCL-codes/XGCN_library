import XGCN
from XGCN.model.base import BaseModel
from XGCN.model.module.mask_neighbor_score import mask_neighbor_score, mask_neighbor_score_user_item
from XGCN.model.module import bpr_loss, bce_loss
from XGCN.data import io, csr
from XGCN.utils.utils import ensure_dir, print_dict, get_formatted_results
from .hashing import ElphHashes

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
import os.path as osp


class LinkPredictor(torch.nn.Module):

    def __init__(self, config, input_dim):
        super(LinkPredictor, self).__init__()
        # self.use_embedding = use_embedding
        # self.use_feature = args.use_feature
        # self.feature_dropout = args.feature_dropout
        # self.label_dropout = args.label_dropout
        # self.dim = args.max_hash_hops * (args.max_hash_hops + 2)
        self.label_lin_layer = Linear(input_dim, input_dim)
        # if args.use_feature:
        #     self.bn_feats = torch.nn.BatchNorm1d(args.hidden_channels)
        # if self.use_embedding:
        #     self.bn_embs = torch.nn.BatchNorm1d(args.hidden_channels)
        self.bn_labels = torch.nn.BatchNorm1d(input_dim)
        # if args.use_feature:
        #     self.lin_feat = Linear(args.hidden_channels,
        #                            args.hidden_channels)
        #     self.lin_out = Linear(args.hidden_channels, args.hidden_channels)
        # out_channels = self.dim + args.hidden_channels if self.use_feature else self.dim
        # if self.use_embedding:
        #     self.lin_emb = Linear(args.hidden_channels,
        #                           args.hidden_channels)
        #     self.lin_emb_out = Linear(args.hidden_channels, args.hidden_channels)
        #     out_channels += args.hidden_channels
        # self.lin = Linear(out_channels, 1)
        self.lin = Linear(input_dim, 1)
        self.p_drop = config['p_drop']

    # def feature_forward(self, x):
    #     """
    #     small neural network applied edgewise to hadamard product of node features
    #     @param x: node features torch tensor [batch_size, 2, hidden_dim]
    #     @return: torch tensor [batch_size, hidden_dim]
    #     """
    #     x = x[:, 0, :] * x[:, 1, :]
    #     # mlp at the end
    #     x = self.lin_out(x)
    #     x = self.bn_feats(x)
    #     x = F.relu(x)
    #     x = F.dropout(x, p=self.feature_dropout, training=self.training)
    #     return x

    # def embedding_forward(self, x):
    #     x = self.lin_emb(x)
    #     x = x[:, 0, :] * x[:, 1, :]
    #     # mlp at the end
    #     x = self.lin_emb_out(x)
    #     x = self.bn_embs(x)
    #     x = F.relu(x)
    #     # x = F.dropout(x, p=self.feature_dropout, training=self.training)

    #     return x

    def forward(self, sf, node_features=None, emb=None):
        sf = self.label_lin_layer(sf)
        sf = self.bn_labels(sf)
        sf = F.relu(sf)
        x = F.dropout(sf, p=self.p_drop, training=self.training)
        # # process node features
        # if self.use_feature:
        #     node_features = self.feature_forward(node_features)
        #     x = torch.cat([x, node_features.to(torch.float)], 1)
        # if emb is not None:
        #     node_embedding = self.embedding_forward(emb)
        #     x = torch.cat([x, node_embedding.to(torch.float)], 1)
        x = self.lin(x)
        return x


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
        # self.mlp = LinkPredictor(self.config, self.get_input_dim()).to(self.device)
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
        ((src, pos, neg), ) = batch_data
        
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
    
    def on_epoch_begin(self):
        self.mlp.train()

    @torch.no_grad()
    def infer_all_target_score(self, src, mask_nei=True):
        target = torch.Tensor.repeat(self.all_target_nodes, len(src))
        _src = torch.repeat_interleave(torch.LongTensor(src), len(self.all_target_nodes))
        links = torch.cat([_src.reshape(-1, 1), target.reshape(-1, 1)], dim=-1)
        
        x = self.get_link_heuristic(links)
        scores = self.mlp(x)
        scores = scores.reshape(-1, len(self.all_target_nodes))
        all_target_score = scores.cpu().numpy()
        
        if mask_nei:
            self.mask_neighbor_score(src, all_target_score)
        
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
        self.mlp.eval()
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
