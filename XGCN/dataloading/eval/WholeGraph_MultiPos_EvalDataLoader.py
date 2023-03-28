from XGCN.data import io

import numpy as np


class WholeGraph_MultiPos_EvalDataLoader:
    
    def __init__(self, file_eval_set, batch_size):
        eval_set = io.load_pickle(file_eval_set)
        self.src = eval_set['src']
        self.pos_list = eval_set['pos_list']
        
        self.batch_size = batch_size
        self.num_total_batch = int(np.ceil(len(self.src) / self.batch_size))
    
    def num_samples(self):
        return len(self.src)
        
    def __len__(self):
        return self.num_total_batch
    
    def __iter__(self):
        self.start = 0
        return self
    
    def __next__(self):
        if self.start >= len(self.src):
            raise StopIteration
        
        src = self.src[self.start : self.start + self.batch_size]
        pos = self.pos_list[self.start : self.start + self.batch_size]
        self.start += self.batch_size
        
        return src, pos
