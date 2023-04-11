import XGCN
from XGCN.data import io, csr
from XGCN.utils.parse_arguments import parse_arguments

import numpy as np
import numba


def _assert_eval_method(eval_method):
    assert eval_method in [
        'one_pos_k_neg',
        'one_pos_whole_graph',
        'multi_pos_whole_graph'
    ]


@numba.jit(nopython=True)
def _generate_strict_neg(src, num_neg, indptr, indices, neg_low, neg_high):
    X = np.empty((len(src), num_neg), dtype=np.int32)
    for i in range(len(src)):
        u = src[i]
        nei = csr.get_neighbors(indptr, indices, u)
        nei = set(nei)
        j = 0
        while j < num_neg:
            while True:
                v = np.random.randint(neg_low, neg_high, (1,))[0]
                if v not in nei:
                    X[i][j] = v
                    j += 1
                    break
    return X


def generate_one_pos_k_neg_eval_set_from_pos_edges(pos_edges, num_neg, info, indptr, indices):
    if info['graph_type'] == 'homo':
        neg_low = 0
        neg_high = info['num_nodes']
    elif info['graph_type'] == 'user-item':
        neg_low = info['num_users']
        neg_high = info['num_users'] + info['num_items']
    src = pos_edges[:,0]
    neg = _generate_strict_neg(src, num_neg, indptr, indices, neg_low, neg_high)
    X = np.concatenate([pos_edges, neg], axis=-1)
    return X


def generate_multi_pos_whole_graph_eval_set_from_pos_edges(pos_edges):
    adj = {}
    for e in pos_edges:
        u, v = e[0], e[1]
        if u in adj:
            adj[u].append(v)
        else:
            adj[u] = [v]

    src = []
    pos_list = []
    for u in adj:
        src.append(u)
        pos_list.append(np.array(adj[u]))
    src = np.array(src)
    
    return {'src': src, 'pos_list': pos_list}


def main():
    
    print("# evaluation_set_generation...")
    
    config = parse_arguments()
    file_input = config['file_input_graph']
    file_output_graph = config['file_output_graph']
    file_output_eval_set = config['file_output_eval_set']
    seed = config['seed']
    
    graph_type = config['graph_type']
    assert graph_type in ['homo', 'user-item']
    
    graph_format = config['graph_format']
    assert graph_format in ['edge_list', 'adjacency_list']
    
    eval_method = config['eval_method']
    _assert_eval_method(eval_method)
    
    num_edge_samples = config['num_edge_samples']

    print("# load graph...")
    if graph_format == 'edge_list':
        E_src, E_dst = io.load_txt_edges(file_input)
    elif graph_format == 'adjacency_list':
        E_src, E_dst = io.load_txt_adj_as_edges(file_input)
    else:
        assert 0
    
    info, indptr, indices = csr.from_edges_to_csr_with_info(
        E_src, E_dst, graph_type
    )
    print("# input graph info:", info)
    
    # guarantee the minimum out-degree of a source node:
    min_src_out_degree = config['min_src_out_degree']
    # guarantee the minimum in-degree of a destination node:
    min_dst_in_degree = config['min_dst_in_degree']
    
    np.random.seed(seed)
    indptr, indices, pos_edges = XGCN.data.split.split_edges(
        indptr, indices, num_edge_samples, min_src_out_degree, min_dst_in_degree
    )
    np.random.shuffle(pos_edges)
    
    print("# generate & save evaluation set...")
    if graph_type == 'user-item':
        pos_edges[:,1] -= info['num_users']  # item ID
    
    if eval_method == 'one_pos_k_neg':
        eval_set = generate_one_pos_k_neg_eval_set_from_pos_edges(
            pos_edges, config['num_neg'], info, indptr, indices
        )
        np.savetxt(file_output_eval_set, eval_set, fmt='%d')
    elif eval_method == 'one_pos_whole_graph':
        eval_set = pos_edges
        np.savetxt(file_output_eval_set, eval_set, fmt='%d')
    elif eval_method == 'multi_pos_whole_graph':
        eval_set = generate_multi_pos_whole_graph_eval_set_from_pos_edges(
            pos_edges
        )
        io.save_adj_eval_set_as_txt(file_output_eval_set, eval_set)
    else:
        assert 0
    
    E_src = csr.get_src_indices(indptr)
    E_dst = indices
    
    if graph_type == 'user-item':
        E_dst -= info['num_users']  # item ID
        
    E = np.stack([E_src, E_dst]).T
    print("# save the graph...")
    np.savetxt(fname=file_output_graph, X=E, fmt='%d')
    
    print("# done!")


if __name__ == '__main__':
    main()
