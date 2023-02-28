from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import ensure_dir
from XGCN.utils import io
from XGCN.utils import csr
from XGCN.utils.csr.CSR_Graph import CSR_Graph_rev_rm_edge

import numpy as np
import os.path as osp


def edges_split(info, indptr, indices, num_sample, min_src_out_degree, min_dst_in_degree):
    '''
        Note: the edges split happens in-place
    '''
    print("# init CSR_Graph_rev_rm_edge...")
    g = CSR_Graph_rev_rm_edge(indptr, indices)
    
    def src_degree_ok(node):
        return (min_src_out_degree < g.out_degree(node))
    
    def dst_degree_ok(node):
        return (min_dst_in_degree < g.in_degree(node))
    
    all_nodes = np.arange(g.num_nodes())
    pos_edges = []
    while True:
        np.random.shuffle(all_nodes)
        exists_ok_node = False
        for s in all_nodes:
            print("sampling edges {}/{} ({:.2f}%)".format(
                len(pos_edges), num_sample, 100*len(pos_edges) / num_sample), end='\r')
            if src_degree_ok(s):
                nei = g.successors(s)
                np.random.shuffle(nei)
                for d in nei:
                    if dst_degree_ok(d):
                        exists_ok_node = True
                        pos_edges.append((s, d))
                        g.remove_successor(s, d)
                        break
                if len(pos_edges) >= num_sample:
                    break
        if len(pos_edges) >= num_sample or not exists_ok_node:
            break
    print("\nnum sampled edges:", len(pos_edges))
    pos_edges = np.array(pos_edges)
    
    print("# csr.to_compact(g.indptr, g.indices)...")
    indptr, indices = csr.to_compact(g.indptr, g.indices)
    info['num_edges'] = len(indices)

    return info, indptr, indices, pos_edges


def main():
    
    config = parse_arguments()
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    
    np.random.seed(config['seed'])
    num_sample = config['num_sample']
    min_src_out_degree = config['min_src_out_degree']
    min_dst_in_degree = config['min_dst_in_degree']
    
    print("# load csr graph...")
    indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
    indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    
    info, indptr, indices, pos_edges = edges_split(
        info, indptr, indices, 
        num_sample, min_src_out_degree, min_dst_in_degree
    )
    
    print("# save...")
    io.save_yaml(osp.join(results_root, 'info.yaml'), info)
    io.save_pickle(osp.join(results_root, 'indptr.pkl'), indptr)
    io.save_pickle(osp.join(results_root, 'indices.pkl'), indices)
    io.save_pickle(osp.join(results_root, 'pos_edges.pkl'), pos_edges)


if __name__ == '__main__':
    
    main()
