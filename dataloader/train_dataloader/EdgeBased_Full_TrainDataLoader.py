from dataloader.utils import generate_one_strict_neg, generate_multi_strict_neg

import torch
import numpy as np


class EdgeBased_Full_TrainDataLoader:
    
    def __init__(self, info, E_src: np.ndarray, E_dst: np.ndarray,
                 batch_size, num_neg=1,
                 ensure_neg_is_not_neighbor=False, csr_indptr=None, csr_indices=None):
        self.dl = torch.utils.data.DataLoader(
            dataset=torch.stack([
                torch.LongTensor(E_src), 
                torch.LongTensor(E_dst)]).T,
            batch_size=batch_size,
            shuffle=True
        )
        self.num_neg = num_neg
        
        if info['dataset_type'] == 'user-item':
            self.neg_low = info['num_users']
            self.neg_high = info['num_users'] + info['num_items']
        else:
            self.neg_low = 0
            self.neg_high = info['num_nodes']
            
        self.ensure_neg_is_not_neighbor = ensure_neg_is_not_neighbor
        if self.ensure_neg_is_not_neighbor:
            self.csr_indptr = csr_indptr
            self.csr_indices = csr_indices
    
    def _generate_strict_neg(self, src):
        if self.num_neg == 1:
            return generate_one_strict_neg(
                self.neg_low, self.neg_high, src, 
                self.csr_indptr, self.csr_indices
            )
        else:
            return generate_multi_strict_neg(
                self.neg_low, self.neg_high, self.num_neg, src, 
                self.csr_indptr, self.csr_indices
            )
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        self.dl_iter = iter(self.dl)
        return self
    
    def __next__(self):
        src_pos = next(self.dl_iter)
        src, pos = src_pos[:,0], src_pos[:,1]
        
        if self.num_neg < 1:
            neg = None
        else:
            if self.ensure_neg_is_not_neighbor:
                neg = torch.LongTensor(self._generate_strict_neg(src.numpy()))
            else:
                neg = torch.randint(self.neg_low, self.neg_high, 
                                    (len(src), self.num_neg)).squeeze()
        
        return src, pos, neg
