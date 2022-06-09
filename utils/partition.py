import numpy as np
import dgl
import os.path as osp


def _get_node_group(num_part, node_map):
    
    node_groups = [[] for _ in range(num_part)]
    for nid, cid in enumerate(node_map):
        node_groups[cid].append(nid)
    
    for i, nids in enumerate(node_groups):
        node_groups[i] = np.array(nids)
    
    return node_groups


def dgl_metis_partition(g, num_part, part_method='metis', results_root='.'):
    assert part_method in ['metis', 'random']
    
    out_path = osp.join(results_root, "dgl_partition_output")
    dgl.distributed.partition_graph(
        g=g,
        graph_name='graph',
        num_parts=num_part,
        out_path=out_path,
        part_method=part_method,
        reshuffle=False  # if True, node_map.npy will not be saved
    )
    node_map = np.load(osp.join(out_path, "node_map.npy"))  # 1d array, node id -> part id
    node_groups = _get_node_group(num_part, node_map)  # list of 1d array: [[nids of part 0], [nids of part 1], ... ]
    return node_map, node_groups
