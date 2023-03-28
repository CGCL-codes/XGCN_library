from ..query import get_neighbors, get_degrees
from ..process import get_reversed, remove_neighbor, to_compact
from ..process import Nan

import numpy as np


class CSR_Graph_rev_rm_edge:
    
    def __init__(self, indptr, indices):
        self.indptr = indptr
        self.indices = indices
        
        self._num_nodes = len(self.indptr) - 1
        self._num_edges = len(self.indices)
        
        self.rev_indptr, self.rev_indices = get_reversed(
            self.indptr, self.indices
        )
        
        self._out_degrees = get_degrees(self.indptr)
        self._in_degrees = get_degrees(self.rev_indptr)
        
        self._is_compact = True
    
    def is_compact(self):
        return self._is_compact
    
    def to_compact(self):
        self.indptr, self.indices = to_compact(self.indptr, self.indices)
        self.rev_indptr, self.rev_indices = to_compact(self.rev_indptr, self.rev_indices)
        self._is_compact = True
    
    def num_nodes(self):
        return self._num_nodes
    
    def num_edges(self):
        return self._num_edges
    
    def _get_neighbors(self, indptr, indices, u):
        x = get_neighbors(indptr, indices, u)
        if self.is_compact():
            return x
        else:
            return x[x != Nan]
    
    def successors(self, u):
        return self._get_neighbors(self.indptr, self.indices, u)
    
    def predecessors(self, u):
        return self._get_neighbors(self.rev_indptr, self.rev_indices, u)
    
    def out_degree(self, u):
        return self._out_degrees[u]
    
    def in_degree(self, u):
        return self._in_degrees[u]
    
    def remove_successor(self, u, v_successor):
        self._is_compact = False
        v = v_successor        
        remove_neighbor(self.indptr, self.indices, src_nid=u, nei_nid=v)
        self._out_degrees[u] -= 1
        remove_neighbor(self.rev_indptr, self.rev_indices, src_nid=v, nei_nid=u)
        self._in_degrees[v] -= 1
    
    def remove_predecessor(self, u, v_predecessor):
        self._is_compact = False
        v = v_predecessor
        remove_neighbor(self.rev_indptr, self.rev_indices, src_nid=u, nei_nid=v)
        self._in_degrees[u] -= 1
        remove_neighbor(self.indptr, self.indices, src_nid=v, nei_nid=u)
        self._out_degrees[v] -= 1
