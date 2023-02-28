from gnn_zoo.utils.parse_arguments import parse_arguments
from gnn_zoo.utils.utils import ensure_dir, print_dict
from gnn_zoo.utils import io
from gnn_zoo.utils import csr

import os.path as osp


def from_edges_to_csr(E_src, E_dst, graph_type):
    assert graph_type in ['homo', 'user-item', ]
    
    info = {}
    info['graph_type'] = graph_type
    
    # assume that num_nodes = max(id) + 1
    if graph_type == 'homo':
        num_nodes = int(max(E_src.max(), E_dst.max()) + 1)
    elif graph_type == 'user-item':
        num_users = int(E_src.max() + 1)
        num_items = int(E_dst.max() + 1)
        info['num_users'] = num_users
        info['num_items'] = num_items
        num_nodes = num_users + num_items
    info['num_nodes'] = num_nodes
    
    if graph_type == 'user-item':
        E_dst += num_users

    print("# from_edges_to_csr ...")
    indptr, indices = csr.from_edges_to_csr(E_src, E_dst, num_nodes)
    _num_edges = len(indices)
    
    print("# remove_repeated_edges ...")
    indptr, indices = csr.remove_repeated_edges_in_csr(indptr, indices)
    num_edges = len(indices)
    print("## {} edges are removed".format(_num_edges - num_edges))
    info['num_edges'] = num_edges

    return info, indptr, indices

    
def main():
    
    config = parse_arguments()
    file_input = config['file_input']
    graph_type = config['graph_type']
    results_root = config['results_root']
    ensure_dir(results_root)
    
    if config['is_adj_list']:
        print("# load txt adj...")
        E_src, E_dst = io.txt_graph.load_adj_as_edges(file_input)
    else:
        print("# load txt edge list...")
        E_src, E_dst = io.txt_graph.load_edges(file_input)
    
    info, indptr, indices = from_edges_to_csr(E_src, E_dst, graph_type)
    
    print("# graph info:")
    print_dict(info)
    
    print("# save...")
    io.save_yaml(osp.join(results_root, 'info.yaml'), info)
    io.save_pickle(osp.join(results_root, 'indptr.pkl'), indptr)
    io.save_pickle(osp.join(results_root, 'indices.pkl'), indices)


if __name__ == '__main__':
    
    main()
