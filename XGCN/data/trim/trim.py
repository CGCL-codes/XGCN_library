import numpy as np


def drop_nodes_randomly(drop_nodes_randomlyg, num_drop, node_mask=None):
    if node_mask is not None:
        all_nid = g.nodes()[node_mask]
    else:
        all_nid = g.nodes()
    np.random.shuffle(all_nid.numpy())
    rm_nid = all_nid[:num_drop]
    g.remove_nodes(rm_nid)


def drop_nodes_by_degree(g, node_mask=None, 
                         min_out_degree=None, max_out_degree=None, 
                         min_in_degree=None, max_in_degree=None):
    if node_mask is not None:
        all_nid = g.nodes()[node_mask]
        out_degrees = g.out_degrees(all_nid)
        in_degrees = g.in_degrees(all_nid)
    else:
        all_nid = g.nodes()
        out_degrees = g.out_degrees()
        in_degrees = g.in_degrees()
    
    masks = []
    if min_out_degree is not None:
        masks.append(out_degrees < min_out_degree)
    if max_out_degree is not None:
        masks.append(out_degrees > max_out_degree)
    if min_in_degree is not None:
        masks.append(in_degrees < min_in_degree)
    if max_in_degree is not None:
        masks.append(in_degrees > max_in_degree)
    assert len(masks) > 0
    
    mask = masks[0]
    for m in masks[1:]:
        mask = mask & m
    
    rm_nid = all_nid[mask]
    g.remove_nodes(rm_nid)
    
    return len(rm_nid)
