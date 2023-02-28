from XGCN.utils import io
from XGCN.utils import csr
from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import ensure_dir

import numpy as np
import os.path as osp
from tqdm import tqdm


def main():
    
    config = parse_arguments()
    
    ppr_data_root = config['ppr_data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    graph_type = config['graph_type']
    
    nei = io.load_pickle(osp.join(ppr_data_root, "nei.pkl"))
    wei = io.load_pickle(osp.join(ppr_data_root, "wei.pkl"))
    
    topk = config['topk']
    nei = nei[:,:topk]
    wei = wei[:,:topk]
    
    # from ppr neighbor to edges
    num_nodes = len(nei)
    max_num_edges = topk * num_nodes
    E_src = np.empty(max_num_edges, dtype=np.int32)
    E_dst = np.empty(max_num_edges, dtype=np.int32)
    
    start = 0
    for u in tqdm(range(num_nodes)): 
        u_nei = nei[u][wei[u] != 0]
        d = len(u_nei)
        u_self = np.full(d, fill_value=u, dtype=np.int32)
        
        E_src[start : start + d] = u_self
        E_dst[start : start + d] = u_nei
        start += d
    E_src = E_src[:start]
    E_dst = E_dst[:start]
    
    # from edges to csr graph
    indptr, indices = csr.from_edges_to_csr(E_src, E_dst, num_nodes)
    num_nodes = len(indptr) - 1
    num_edges = len(indices)
    info = {
        'graph_type': graph_type,
        'num_nodes': num_nodes,
        'num_edges': num_edges
    }

    io.save_yaml(osp.join(results_root, 'info.yaml'), info)
    io.save_pickle(osp.join(results_root, 'indptr.pkl'), indptr)
    io.save_pickle(osp.join(results_root, 'indices.pkl'), indices)


if __name__ == '__main__':
    
    main()
