import XGCN
from XGCN.data import io, csr
from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import ensure_dir

import os.path as osp


def main():
    
    print("# process_int_graph...")
    
    config = parse_arguments()
    file_input = config['file_input']
    data_root = config['data_root']
    ensure_dir(data_root)
    
    graph_type = config['graph_type']
    assert graph_type in ['homo', 'user-item']
    
    graph_format = config['graph_format']
    assert graph_format in ['edge_list', 'adjacency_list']
    
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
    
    print("# save...")
    io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    io.save_pickle(osp.join(data_root, 'indptr.pkl'), indptr)
    io.save_pickle(osp.join(data_root, 'indices.pkl'), indices)

    print("# done!")


if __name__ == '__main__':
    main()
