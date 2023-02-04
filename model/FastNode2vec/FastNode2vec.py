import numpy
import numpy as np
import scipy
from tqdm import tqdm
import gensim
import os.path as osp

from .RandomWalkGraph import RandomWalkGraph


def get_sentences(walk_graph: RandomWalkGraph, num_walks, walk_length, p, q, walk_seed=None):
    if walk_seed is not None:
        np.random.seed(walk_seed)  # generate the same sentences for each epoch
    nids = np.arange(walk_graph.num_nodes)
    while num_walks > 0:
        num_walks -= 1
        np.random.shuffle(nids)
        for nid in nids:            
            ##--
            if walk_graph.indptr[nid] < walk_graph.indptr[nid + 1]: 
                yield walk_graph.generate_random_walk(walk_length, p, q, start=nid).tolist()


class FastNode2vec:
    """
        a simple wrapper for fastnode2vec
        https://louisabraham.github.io/articles/node2vec-sampling.html
        https://github.com/louisabraham/fastnode2vec
    """
    def __init__(self, indptr, indices):
        self.walk_graph = RandomWalkGraph(indptr, indices)
        self.num_nodes = len(indptr) - 1
        self.node_degree = indptr[1:] - indptr[:-1]
        self.embs = None

    def run_node2vec(self, dim, epochs=1, alpha=0.001, min_alpha=0.001, alpha_schedule=None, num_walks=20, walk_length=30, window=3, 
                     p=1.0, q=1.0, num_neg=5, walk_seed=None, callbacks=[]):
        if alpha_schedule is not None:
            epoch_list = alpha_schedule[0]
            alpha_list = alpha_schedule[1]
            assert epoch_list[-1] >= epochs + 1
            fn_alpha = scipy.interpolate.interp1d(epoch_list, alpha_list, kind='linear')
        
        def _get_sentences():
            return get_sentences(self.walk_graph, num_walks, walk_length, p, q, walk_seed)
        
        class SentencesWapper:
            
            def __init__(self, get_sentences, epochs, length):
                self.get_sentences = get_sentences
                self.epochs = epochs
                self.epoch = 0
                self.length = length
                
            def __iter__(self):
                if self.epoch == 0: 
                    desc = ""
                else:
                    desc = "epoch {}/{}".format(self.epoch, self.epochs)
                self.epoch += 1
                return iter(tqdm(self.get_sentences(), total=self.length, desc=desc))
            
        sent_wapper = SentencesWapper(_get_sentences, epochs=epochs, length=self.num_nodes * num_walks)
        
        self.model = gensim.models.Word2Vec(
            vector_size=dim, window=window, negative=num_neg, 
            epochs=epochs, 
            alpha=alpha, 
            min_alpha=min_alpha,
            min_count=1, workers=6, seed=1
        )
        
        print(">> build vocab...")
        self.model.build_vocab(sent_wapper)
        # self.model.build_vocab_from_freq(self.node_degree) ##--
        
        print(">> train...")
        if alpha_schedule is not None:
            for epoch in range(1, epochs + 1):
                start_alpha = fn_alpha(epoch).item()
                end_alpha = fn_alpha(epoch + 1).item()
                print(">> start_alpha {}, end_alpha {}".format(start_alpha, end_alpha))
                self.model.train(sent_wapper, epochs=1, start_alpha=start_alpha, end_alpha=end_alpha,
                            total_examples=self.model.corpus_count, callbacks=callbacks,
                            total_words=self.num_nodes
                            )
        else:  # use default alpha schedule
            self.model.train(sent_wapper, epochs=self.model.epochs,
                        total_examples=self.model.corpus_count, callbacks=callbacks)
        
        
    def get_embeddings(self):
        self.embs = self.model.wv[list(range(self.num_nodes))]
        return self.embs

    def save_word2vec_format(self, filename):
        self.self.model.wv.save_word2vec_format(filename)
