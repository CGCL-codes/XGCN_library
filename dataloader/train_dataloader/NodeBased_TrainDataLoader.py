from dataloader.utils import generate_pos_and_one_strict_neg, generate_src_pos_and_one_strict_neg

import torch
import numpy as np


class NodeBased_TrainDataLoader:
    
    def __init__(self, info, num_edges_per_epoch, batch_size, 
                 train_csr_indptr, train_csr_indices):
        # same as the dataloader used in LightGCN paper
        
        self.src_low = 0
        if info['dataset_type'] == 'user-item':
            num_users = info['num_users']
            num_items = info['num_items']
            self.src_high = num_users
            self.neg_low = num_users
            self.neg_high = num_users + num_items
        else:
            num_nodes = info['num_nodes']
            self.src_high = num_nodes
            self.neg_low = 0
            self.neg_high = num_nodes

        self.num_edges_per_epoch = num_edges_per_epoch
        self.batch_size = batch_size
        self.train_csr_indptr = train_csr_indptr
        self.train_csr_indices = train_csr_indices
        
        self.batch_per_epoch = int(np.ceil(self.num_edges_per_epoch / self.batch_size))
    
    def __len__(self):
        return self.batch_per_epoch
    
    def __iter__(self):
        self.batch_remain = self.batch_per_epoch
        self.num_edges_remain = self.num_edges_per_epoch
        return self
    
    def __next__(self):
        if self.batch_remain == 0:
            raise StopIteration
        self.batch_remain -= 1
        
        if self.num_edges_remain > self.batch_size:
            _batch_size = self.batch_size
        else:
            _batch_size = self.num_edges_remain
        self.num_edges_remain -= self.batch_size
        
        # src = np.random.randint(0, self.src_high, _batch_size)
        # pos, neg = generate_pos_and_one_strict_neg(
        #     self.neg_low, self.neg_high, src,
        #     self.train_csr_indptr,
        #     self.train_csr_indices
        # )
        src, pos, neg = generate_src_pos_and_one_strict_neg(
            self.src_low, self.src_high,
            self.neg_low, self.neg_high, self.batch_size,
            self.train_csr_indptr,
            self.train_csr_indices
        )

        return torch.LongTensor(src), torch.LongTensor(pos), torch.LongTensor(neg)
