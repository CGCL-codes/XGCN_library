import numpy as np


class EpochReducer(object):
    
    def __init__(self, sample_indices: np.ndarray, ratio):
        self.sample_indices = sample_indices
        self.num_total_samples = len(self.sample_indices)
        self.num_epoch_samples = int(self.num_total_samples * ratio)
        
        self._shuffle_and_reset()
    
    def _shuffle_and_reset(self):
        np.random.shuffle(self.sample_indices)
        self.start = 0
        self.end = self.num_epoch_samples
        
    def get_sample_indices_for_this_epoch(self):
        loop_end = self.end + self.num_epoch_samples > self.num_total_samples
        if loop_end:
            self.end = None
        
        epoch_sample_indices = self.sample_indices[self.start : self.end]
        
        if loop_end:
            self._shuffle_and_reset()
        else:
            self.start = self.end
            self.end += self.num_epoch_samples
        
        return epoch_sample_indices


class BatchIndexer(object):
    
    def __init__(self, X, batch_size):
        self.X = X
        self.batch_size = batch_size
        self.num_batch_per_epoch = int(len(self.X) / batch_size)
    
    def __len__(self):
        return self.num_batch_per_epoch
    
    def __getitem__(self, batch_idx):
        start = self.batch_size * batch_idx
        end = start + self.batch_size
        if end + self.batch_size > len(self.X):
            end = None
        
        return self.X[start : end]
