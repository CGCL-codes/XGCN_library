from .Node2vecRandomWalkGraph import Node2vecRandomWalkGraph

import numpy as np


class Node2vecRandomWalker:
    
    def __init__(self, indptr, indices, 
                 num_walks=16, walk_length=16, p=1.0, q=1.0):
        self.num_walks = num_walks
        self.num_nodes = len(indptr) - 1
        self.num_sentences = self.num_walks * self.num_nodes
        
        self.nids = np.arange(self.num_nodes)
        self.walk_graph = Node2vecRandomWalkGraph(
            indptr, indices, walk_length, p, q
        )
    
    def __len__(self):
        return self.num_sentences
    
    def __iter__(self):
        # self.remain_walks = self.num_walks
        # self.remain_nodes = self.num_nodes
        # return self
        remain_walks = self.num_walks
        while remain_walks > 0:
            remain_walks -= 1
            np.random.shuffle(self.nids)
            for nid in self.nids:
                if self.walk_graph.indptr[nid] < self.walk_graph.indptr[nid + 1]: 
                    yield self.walk_graph.generate_random_walk(start_node=nid).tolist()

    # def __next__(self):
    #     pass

# def get_walks(walk_graph: RandomWalkGraph, num_walks, walk_length, p, q, walk_seed=None):
#     nids = np.arange(walk_graph.num_nodes)
