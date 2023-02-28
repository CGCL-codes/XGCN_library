from .base import BaseDataset, NodeListDataset
from .base import BaseSampler, BatchSampleIndicesGenerator
from XGCN.utils.utils import get_unique

import torch
import dgl


class DataLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset: BaseDataset, num_workers=0):
        self.dataset = dataset
        super().__init__(
            dataset=self.dataset,
            batch_size=None, shuffle=False,
            num_workers=num_workers
        )
    
    def __iter__(self):
        self.dataset.on_epoch_start()
        return super().__iter__()


class LinkDataset(NodeListDataset):
    
    def __init__(self,
                 pos_sampler: BaseSampler,
                 neg_sampler: BaseSampler,
                 batch_sample_indices_generator: BatchSampleIndicesGenerator):
        self.pos_sampler = pos_sampler
        self.neg_sampler = neg_sampler
        self.batch_sample_indices_generator = batch_sample_indices_generator

    def __len__(self):
        return len(self.batch_sample_indices_generator)
    
    def __getitem__(self, batch_idx):
        batch_sample_indices = self.batch_sample_indices_generator[batch_idx]
        
        src, pos = self.pos_sampler(batch_sample_indices)
        neg = self.neg_sampler({'src': src, 'pos': pos})
        
        node_list = [src, pos, neg]
        return [node_list, ]
    
    def on_epoch_start(self):
        self.batch_sample_indices_generator.on_epoch_start()


class BlockDataset(BaseDataset):
    
    def __init__(self, g, block_sampler,
                 dataset: NodeListDataset):
        self.g = g
        self.block_sampler = block_sampler
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, batch_idx):
        re = self.dataset[batch_idx]
        node_list = re[0]
        
        flat_batch_nid = torch.cat([nid.flatten() for nid in node_list])
        
        unique_nid, idx_mapping = get_unique(flat_batch_nid)
        # unique_nid: 1d-tensor, remove repeated nid in batch_nid
        # idx_mapping: map a nid into the index of the unique_nid tensor

        input_nid, output_nid, blocks = self.block_sampler.sample_blocks(
            self.g, unique_nid.to(self.g.device), exclude_eids=None
        )
        
        return re, (input_nid, output_nid, blocks, idx_mapping)

    def on_epoch_start(self):
        self.dataset.on_epoch_start()


class ClusterDataLoader:
    
    def __init__(self, g, partition_cache_filepath,
                 num_parts, group_size, batch_size, num_workers, subgraph_device):
        
        #  g : DGLGraph. The original graph. Must be homogeneous and on CPU.
        sampler = dgl.dataloading.ClusterGCNSampler(
            g, num_parts, cache_path=partition_cache_filepath
        )
        self.subgraph_dl = dgl.dataloading.DataLoader(
            g, torch.arange(num_parts), sampler,
            batch_size=group_size,
            shuffle=True, drop_last=False, num_workers=num_workers
        )
        
        self.subgraph_device = subgraph_device
        self.subgraph_train_dl = SubgraphTrainDataLoader(batch_size)
        
    def __len__(self):
        return len(self.subgraph_dl)
    
    def __iter__(self):
        return self.batch_data_generator()
    
    def batch_data_generator(self):
        for subg in self.subgraph_dl:
            subg = subg.to(self.subgraph_device)
            
            self.subgraph_train_dl.set_graph(subg)
            
            for batch_data in self.subgraph_train_dl:
                yield subg, batch_data

    def enable_cpu_affinity(self):
        self.subgraph_dl.enable_cpu_affinity()


class SubgraphTrainDataLoader:
    
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.g = None
        
    def set_graph(self, g):
        self.g = g

    def __iter__(self):
        self.E_src, self.E_dst = self.g.edges()
        dl = torch.utils.data.DataLoader(
            torch.arange(len(self.E_src)), batch_size=self.batch_size, shuffle=True
        )
        self.dl_iter = iter(dl)
        return self
    
    def __next__(self):
        idx = next(self.dl_iter)
        return self.E_src[idx], self.E_dst[idx], torch.randint(0, self.g.num_nodes(), (len(idx), ))
