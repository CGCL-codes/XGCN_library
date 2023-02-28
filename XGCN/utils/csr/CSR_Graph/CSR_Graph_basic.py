from ..query import get_neighbors, get_degrees
from ..process import get_src_indices

import numpy as np


class CSR_Graph_basic:
    
    def __init__(self, indptr, indices):
        self.indptr = indptr
        self.indices = indices
        self.src_indices = None
        
        self._nodes = None
        self._degrees = None
        
    def num_nodes(self):
        return len(self.indptr) - 1
    
    def num_edges(self):
        return len(self.indices)
    
    def nodes(self):
        if self._nodes is None:
            self._nodes = np.arange(self.num_nodes())
        return self._nodes

    def edges(self):
        if self.src_indices is None:
            self.src_indices = get_src_indices(self.indptr)
        return (self.src_indices, self.indices)
    
    def degrees(self):
        if self._degrees is None:
            self._degrees = get_degrees(self.indptr)
        return self._degrees

    def neighbors(self, u):
        return get_neighbors(self.indptr, self.indices, u)
