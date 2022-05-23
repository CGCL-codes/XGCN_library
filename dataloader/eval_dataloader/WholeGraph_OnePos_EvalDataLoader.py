import numpy as np
import torch


class WholeGraph_OnePos_EvalDataLoader:
    
    def __init__(self, pos_edges,
                 batch_size, model=None, mask_neighbor_score=None):
        self._num_samples = len(pos_edges)
        self.dl = torch.utils.data.DataLoader(
            dataset=pos_edges,
            batch_size=batch_size
        )
        self.model = model
        self.mask_neighbor_score = mask_neighbor_score
    
    def num_samples(self):
        return self._num_samples
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        if self.model is not None:
            self.model.one_pos = True
            self.model.eval_on_whole_graph = True
            self.model.mask_nei = self.mask_neighbor_score
        self.dl_iter = iter(self.dl)
        return self
    
    def __next__(self):
        src_pos = next(self.dl_iter)
        src, pos = src_pos[:,0].numpy(), src_pos[:,1].numpy()  # return np.ndarray when eval
        return src, pos
