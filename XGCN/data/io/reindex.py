from XGCN.utils.ReIndexDict import ReIndexDict
from XGCN.utils.utils import wc_count

import numpy as np
from tqdm import tqdm


def load_and_reindex_homogeneous_graph(filename, skiprows=0, delimiter=None):
    id_mapping = ReIndexDict()
    num_edges = wc_count(filename) - skiprows
    E_src = np.empty(num_edges, dtype=np.uint32)
    E_dst = np.empty(num_edges, dtype=np.uint32)    
    with open(filename, 'r') as f:
        for _ in range(skiprows):
            f.readline()
        for i in tqdm(range(num_edges)):
            s = f.readline().split(delimiter)
            raw_src, raw_dst = int(s[0]), int(s[1])
            E_src[i] = id_mapping[raw_src]
            E_dst[i] = id_mapping[raw_dst]
    return E_src, E_dst, id_mapping


def load_and_reindex_user_item_graph(filename, skiprows=0, delimiter=None):
    user_id_mapping = ReIndexDict()
    item_id_mapping = ReIndexDict()
    num_edges = wc_count(filename) - skiprows
    E_src = np.empty(num_edges, dtype=np.uint32)
    E_dst = np.empty(num_edges, dtype=np.uint32)
    with open(filename, 'r') as f:
        for _ in range(skiprows):
            f.readline()
        for i in tqdm(range(num_edges)):
            s = f.readline().split(delimiter)
            raw_src, raw_dst = int(s[0]), int(s[1])
            E_src[i] = user_id_mapping[raw_src]
            E_dst[i] = item_id_mapping[raw_dst]
    return E_src, E_dst, user_id_mapping, item_id_mapping
