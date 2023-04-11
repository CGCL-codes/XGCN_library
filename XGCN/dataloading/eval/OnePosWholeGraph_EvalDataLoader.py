from XGCN.data import io

import torch


class OnePosWholeGraph_EvalDataLoader:
    
    def __init__(self, file_eval_set, batch_size):
        pos_edges = io.load_pickle(file_eval_set)
        self._num_samples = len(pos_edges)
        self.dl = torch.utils.data.DataLoader(
            dataset=pos_edges,
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
        src_pos = next(self.dl_iter)
        src, pos = src_pos[:,0].numpy(), src_pos[:,1].numpy()  # return np.ndarray when eval
        return src, pos
