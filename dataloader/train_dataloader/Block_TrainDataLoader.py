from dataloader.train_dataloader import EdgeBased_Sampling_TrainDataLoader
from utils.utils import element_wise_map

import numpy as np
import torch
import dgl.dataloading as dgldl


class _EdgeBased_EdgeSampler(torch.utils.data.Dataset):
    
    def __init__(self, info, E_src: np.ndarray, E_dst: np.ndarray,
                 batch_size, num_neg=1, ratio=0.1):
        # for each epoch, sample ratio*num_edges edges from all edges
        self.E_src = E_src
        self.E_dst = E_dst
        self.num_edges = len(E_src)
        
        if info['dataset_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
        
        self.batch_size = batch_size
        self.num_neg = num_neg
        
        self.batch_per_epoch = int(self.num_edges * ratio / self.batch_size)
        self.batch_remain = None
    
    def __len__(self):
        return self.batch_per_epoch
    
    def __getitem__(self, idx):
        eid = torch.randint(0, self.num_edges, (self.batch_size,)).numpy()
        
        src = torch.LongTensor(self.E_src[eid])
        pos = torch.LongTensor(self.E_dst[eid])
        neg = torch.randint(self.neg_low, self.neg_high, (len(src), self.num_neg)).squeeze()
        
        return torch.cat([src, pos, neg])


class EdgeBased_Sampling_Block_TrainDataLoader:

    def __init__(self,
            info, E_src, E_dst,
            batch_size, 
            num_neg=1, ratio=0.1,
            node_collate_graph=None, num_gcn_layer=1, num_layer_sample=[],
            num_workers=0
        ):
        if len(num_layer_sample) != 0:  # use neighbor sampling
            assert num_gcn_layer == len(num_layer_sample)
            block_sampler = dgldl.MultiLayerNeighborSampler(
                num_layer_sample  # e.g. [10, 10]
            )
        else:
            block_sampler = dgldl.MultiLayerFullNeighborSampler(num_gcn_layer)
        
        self.edge_sampler = _EdgeBased_EdgeSampler(
            info, E_src, E_dst,
            batch_size, num_neg, ratio
        )

        self.node_collator = dgldl.NodeCollator(
            node_collate_graph,
            node_collate_graph.nodes(),
            block_sampler
        )

        def collate_fn(batch_nids):
            batch_nids = batch_nids[0]
            unique_batch_nids = list(set(batch_nids.tolist()))
            
            nid_mapping = {nid: i for i, nid in enumerate(unique_batch_nids)}
            
            def index_mapping(nid):
                return nid_mapping[nid]

            def handle_idx(nids):
                return element_wise_map(index_mapping, nids)
            
            local_idx = handle_idx(batch_nids)
            
            input_nids, output_nids, blocks = self.node_collator.collate(unique_batch_nids)
            
            return batch_nids, local_idx, input_nids, output_nids, blocks

        self.dl = torch.utils.data.DataLoader(
            dataset=self.edge_sampler,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=num_workers  # ``num_workers=0`` means that the data will be loaded in the main process
        )
        
    def __len__(self):
        return len(self.edge_sampler)
    
    def __iter__(self):
        return iter(self.dl)
