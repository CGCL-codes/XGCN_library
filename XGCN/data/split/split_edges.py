from XGCN.data import csr

import numpy as np
import torch
import dgl


def split_edges(indptr, indices, num_sample, min_src_out_degree, min_dst_in_degree):
    '''
        Note: the edges split happens in-place
    '''
    g = csr.CSR_Graph_rev_rm_edge(indptr, indices)
    
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
            if src_degree_ok(s):
                nei = g.successors(s)
                np.random.shuffle(nei)
                for d in nei:
                    if dst_degree_ok(d):
                        print("sampling edges {}/{} ({:.2f}%)".format(
                            len(pos_edges), num_sample, 100*len(pos_edges) / num_sample), end='\r')
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
    
    indptr, indices = csr.to_compact(g.indptr, g.indices)
    
    return indptr, indices, pos_edges
