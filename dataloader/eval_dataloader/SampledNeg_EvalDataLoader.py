import numpy as np
import torch


class SampledNeg_EvalDataLoader:
    
    def __init__(self, src_pos_neg: np.ndarray, batch_size, model=None):
        self._num_samples = len(src_pos_neg)  # [[src, pos, neg1, neg2, ...], ]
        self.dl = torch.utils.data.DataLoader(
            dataset=src_pos_neg,
            batch_size=batch_size
        )
        self.model = model

    def num_samples(self):
        return self._num_samples
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        if self.model is not None:
            self.model.one_pos = True
            self.model.eval_on_whole_graph = False
            self.model.mask_nei = False
        self.dl_iter = iter(self.dl)
        return self
    
    def __next__(self):
        src_pos_neg = next(self.dl_iter)
        src, pos, neg = src_pos_neg[:,0], src_pos_neg[:,1], src_pos_neg[:,2:]
        return src.numpy(), pos.numpy(), neg.numpy()
