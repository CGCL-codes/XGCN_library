import numpy as np


class WholeGraph_MultiPos_EvalDataLoader:
    
    def __init__(self, src: np.ndarray, pos_list: list, 
                 batch_size, model=None, mask_neighbor_score=None):
        self.src = src
        self.pos_list = pos_list
        self.batch_size = batch_size
        self.num_total_batch = int(np.ceil(len(pos_list) / batch_size))
        self.model = model
        self.mask_neighbor_score = mask_neighbor_score
    
    def num_samples(self):
        return len(self.src)
        
    def __len__(self):
        return self.num_total_batch
    
    def __iter__(self):
        if self.model is not None:
            self.model.one_pos = False
            self.model.eval_on_whole_graph = True
            self.model.mask_nei = self.mask_neighbor_score
        self.start = 0
        return self
    
    def __next__(self):
        if self.start >= len(self.src):
            raise StopIteration
        
        src = self.src[self.start : self.start + self.batch_size]
        pos = self.pos_list[self.start : self.start + self.batch_size]
        self.start += self.batch_size
        
        return src, pos
