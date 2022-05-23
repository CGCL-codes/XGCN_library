from utils.utils import ensure_dir, print_dict
from utils import io
from data.csr_graph_helper import get_src_indices, from_edges_to_csr, remove_repeated_edges_in_csr

import numpy as np
from scipy.sparse import csr_matrix
import os.path as osp


def handle_train_graph(E_src, E_dst,
                       data_root, dataset_name, dataset_type):
    assert dataset_type in ['user-item', 'social']
    
    ensure_dir(data_root)
    
    info = {}
    info['dataset_name'] = dataset_name
    info['dataset_type'] = dataset_type
    
    print("## infer dataset info ...")
    # assume that num_nodes = max(id) + 1
    if dataset_type == 'user-item':
        num_users = int(E_src.max() + 1)
        num_items = int(E_dst.max() + 1)
        info['num_users'] = num_users
        info['num_items'] = num_items
        num_nodes = num_users + num_items
    else:
        num_nodes = int(max(E_src.max(), E_dst.max()) + 1)
    info['num_nodes'] = num_nodes
    
    if dataset_type == 'user-item':
        E_dst += num_users
    
    print("## construct directed csr graph ...")
    ###################################
    # use scipy.sparse.csr_matrix to construct csr graph:
    # _data = np.empty(len(E_src), dtype=np.int64)
    # csr_g = csr_matrix((_data, (E_src, E_dst)), shape=(num_nodes, num_nodes))  # repeated edges will be removed
    # indptr = csr_g.indptr
    # indices = csr_g.indices
    ###################################
    
    print("- from_edges_to_csr ...")
    indptr, indices = from_edges_to_csr(E_src, E_dst, num_nodes)
    
    del E_src, E_dst
    
    print("- remove_repeated_edges_in_csr ...")
    indptr, indices = remove_repeated_edges_in_csr(indptr, indices)
    
    print("- get_src_indices ...")
    src_indices = get_src_indices(indptr)
    
    info['num_edges'] = len(indices)
    print("- graph info:")
    print_dict(info)
    io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    
    print("## save directed csr graph ...")
    io.save_pickle(osp.join(data_root, 'train_csr_indptr.pkl'), indptr)
    io.save_pickle(osp.join(data_root, 'train_csr_indices.pkl'), indices)
    io.save_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'), src_indices)

    print("## construct undirected csr graph ...")
    undi_E_src = np.concatenate([src_indices, indices])
    undi_E_dst = np.concatenate([indices, src_indices])
    
    del indptr, indices, src_indices
    
    print("- from_edges_to_csr ...")
    undi_indptr, undi_indices = from_edges_to_csr(undi_E_src, undi_E_dst, num_nodes)
    
    del undi_E_src, undi_E_dst
    
    print("- remove_repeated_edges_in_csr ...")
    undi_indptr, undi_indices = remove_repeated_edges_in_csr(undi_indptr, undi_indices)
    
    print("- get_src_indices ...")
    undi_src_indices = get_src_indices(undi_indptr)
    
    # undi_csr_g = csr_matrix((_data, (undi_E_src, undi_E_dst)), shape=(num_nodes, num_nodes))
    # undi_indptr = undi_csr_g.indptr
    # undi_indices = undi_csr_g.indices
    # del undi_csr_g

    print("## save undirected csr graph ...")
    io.save_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'), undi_indptr)
    io.save_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'), undi_indices)
    io.save_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'), undi_src_indices)

    print("## done!")
