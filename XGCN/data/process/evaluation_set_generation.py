import XGCN
from XGCN.data import io, csr
from XGCN.utils.parse_arguments import parse_arguments

import numpy as np


def _assert_evaluation_method(evaluation_method):
    assert evaluation_method in [
        'one_pos_k_neg',
        'one_pos_whole_graph',
        'multi_pos_whole_graph'
    ]


def main():
    
    print("# evaluation_set_generation...")
    
    config = parse_arguments()
    seed = config['seed']
    file_input = config['file_input']
    
    graph_type = config['graph_type']
    assert graph_type in ['homo', 'user-item']
    
    graph_format = config['graph_format']
    assert graph_format in ['edge_list', 'adjacency_list']
    
    num_edges_for_val = config['num_edges_for_val']
    num_edges_for_test = config['num_edges_for_test']
    num_edge_samples = num_edges_for_val + num_edges_for_test
    
    # guarantee the minimum out-degree of a source node:
    min_src_out_degree = config['min_src_out_degree']
    # guarantee the minimum in-degree of a destination node:
    min_dst_in_degree = config['min_dst_in_degree']
    
    val_method = config['val_method']
    _assert_evaluation_method(val_method)
    test_method = config['test_method']
    _assert_evaluation_method(test_method)
    
    file_output = config['file_output_val_set']
    file_output = config['file_output_set_set']

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
    
    np.random.seed(seed)
    pos_edges = XGCN.data.split.split_edges(
        indptr, indices, num_edge_samples, min_src_out_degree, min_dst_in_degree
    )
    
    
    print("# done!")


if __name__ == '__main__':
    main()
