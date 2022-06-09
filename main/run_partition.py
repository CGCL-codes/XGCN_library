import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from utils.partition import dgl_metis_partition

import dgl
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)

    # partition the directed graph
    E_src = io.load_pickle(osp.join(data_root, 'train_csr_src_indices.pkl'))
    E_dst = io.load_pickle(osp.join(data_root, 'train_csr_indices.pkl'))
    g = dgl.graph((E_src, E_dst))

    node_map, node_groups = dgl_metis_partition(
        g, 
        num_part=config['num_part'], 
        part_method=config['part_method'],  # metis or random
        results_root=results_root
    )
    
    io.save_pickle(osp.join(results_root, 'node_map.pkl'), node_map)
    io.save_pickle(osp.join(results_root, 'node_groups.pkl'), node_groups)


if __name__ == "__main__":
    
    setproctitle.setproctitle('partition-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
