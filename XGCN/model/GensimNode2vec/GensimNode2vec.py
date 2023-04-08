from XGCN.dataloading.random_walk import Node2vecRandomWalker
from XGCN.model.base import BaseEmbeddingModel
from XGCN.data import io, csr

from gensim.models import Word2Vec
import torch
import os.path as osp
from tqdm import tqdm


class GensimNode2vec(BaseEmbeddingModel):
    
    def __init__(self, config, data):
        super().__init__(config, data)
        
        # load random walk csr graph
        print("load csr graph...")
        data_root = self.config['data_root']
        indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
        indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
        data['indptr'] = indptr
        data['indices'] = indices
        
        indptr, indices = csr.get_undirected(indptr, indices)
        
        # init Node2vecRandomWalker
        self.walker = Node2vecRandomWalker(
            indptr, indices,
            num_walks=self.config['num_walks'],
            walk_length=self.config['walk_length'],
            p=self.config['p'], q=self.config['q']
        )
        
        class SentencesWapper:
            
            def __init__(self, walker):
                self.walker = walker
            
            def __iter__(self):
                return iter(tqdm(self.walker))

        self.sentences = SentencesWapper(self.walker)
        
        # init Gensim Word2vec
        self.model = Word2Vec(
            vector_size=self.config['emb_dim'],
            window=self.config['context_size'],
            negative=self.config['num_neg'],
            workers=self.config['num_workers'],
            min_count=1,
        )
        
        print("build vocab...")
        self.model.create_vocab(self.sentences)
        # self.model.create_vocab_from_freq(indptr[1:] - indptr[:-1])

    def train_an_epoch(self):
        lr = self.config['emb_lr']
        self.model.train(
            corpus_iterable=self.sentences,
            total_examples=self.model.corpus_count,
            total_words=self.info['num_nodes'],
            epochs=1,
            start_alpha=lr, end_alpha=lr,
            compute_loss=True
        )
        loss = self.model.get_latest_training_loss()
        return loss
    
    def on_eval_begin(self):
        self.out_emb_table = torch.FloatTensor(
            self.model.wv[list(range(self.info['num_nodes']))]
        )
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users']:]
        else:
            self.target_emb_table = self.out_emb_table
