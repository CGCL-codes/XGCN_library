from utils import io
from dataloader.train_dataloader import \
    EdgeBased_Sampling_TrainDataLoader, EdgeBased_Sampling_Block_TrainDataLoader, \
    EdgeBased_Full_TrainDataLoader, \
    NodeBased_TrainDataLoader

import torch
import dgl
import os.path as osp


def build_train_dl(config, data):
    if config['model'] == 'node2vec':
        train_dl = _build_node2vec_train_dl(config, data)
    else:
        train_dl = _build_gnn_train_dl(config, data)
    return train_dl


def _build_node2vec_train_dl(config, data):
    model = data['model']
    train_dl = model.model.loader(
        batch_size=config['train_batch_size'],
        shuffle=True, num_workers=config['num_workers']
    )
    return train_dl


def _build_gnn_train_dl(config, data):
    data_root = config['data_root']
    info = data['info']
    
    batch_size = config['train_batch_size']
    num_neg = config['num_neg']
    ensure_neg_is_not_neighbor = config['ensure_neg_is_not_neighbor']
    
    indptr = io.load_pickle(osp.join(data_root, 'train_csr_indptr.pkl'))
    src_indices = io.load_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    
    E_src = src_indices
    E_dst = indices
    
    if 'edge_sample_ratio' in config:
        ratio = config['edge_sample_ratio']
    else:
        ratio = 0.1
    
    if config['train_dl'] == 'EdgeBased_Sampling_TrainDataLoader':
        if 'use_degree_for_neg_sample' in config and config['use_degree_for_neg_sample']:
            use_degree_for_neg_sample = True
        else:
            use_degree_for_neg_sample = False
        
        train_dl = EdgeBased_Sampling_TrainDataLoader(
            info, E_src, E_dst,
            batch_size=batch_size, num_neg=num_neg, ratio=ratio,  # for each epoch, sample ratio*num_edges edges from all edges
            ensure_neg_is_not_neighbor=ensure_neg_is_not_neighbor, 
            csr_indptr=indptr, csr_indices=indices, use_degree_for_neg_sample=use_degree_for_neg_sample
        )
    
    elif config['train_dl'] == 'EdgeBased_Full_TrainDataLoader':
        E_src = io.load_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'))
        E_dst = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
        
        train_dl = EdgeBased_Full_TrainDataLoader(
            info, E_src, E_dst,
            batch_size=batch_size, num_neg=num_neg,
            ensure_neg_is_not_neighbor=ensure_neg_is_not_neighbor, 
            csr_indptr=indptr, csr_indices=indices
        )
    
    elif config['train_dl'] == 'NodeBased_TrainDataLoader':
        train_dl = NodeBased_TrainDataLoader(
            info=info, num_edges_per_epoch=int(info['num_edges'] * ratio),
            batch_size=batch_size,
            train_csr_indptr=indptr, train_csr_indices=indices
        )
    
    elif config['train_dl'] == 'EdgeBased_Sampling_Block_TrainDataLoader':
        # build node_collate_graph
        g_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
        g_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
        node_collate_graph = dgl.graph((g_src, g_dst))

        train_dl = EdgeBased_Sampling_Block_TrainDataLoader(
            info, E_src, E_dst,
            batch_size,
            num_neg, ratio,
            node_collate_graph,
            num_gcn_layer=config['num_gcn_layers'],
            num_layer_sample=eval(config['num_layer_sample']),
            num_workers=config['num_workers']
        )

        data['node_collate_graph'] = node_collate_graph
        data['node_collator'] = train_dl.node_collator

    else:
        assert 0
    
    return train_dl
