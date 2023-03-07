from XGCN.utils import io

import torch


class OnePosKNeg_EvalDataLoader:
    
    def __init__(self, file_eval_set, batch_size):
        X = io.load_pickle(file_eval_set)
        self._num_samples = len(X)
        self.dl = torch.utils.data.DataLoader(
            dataset=X,
            batch_size=batch_size
        )

    def num_samples(self):
        return self._num_samples
    
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        self.dl_iter = iter(self.dl)
        return self
    
    def __next__(self):
        x = next(self.dl_iter)
        src, pos, neg = x[:,0].numpy(), x[:,1].numpy(), x[:,2:].numpy()
        return src, pos, neg
