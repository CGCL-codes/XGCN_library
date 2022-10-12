import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from model.BaseEmbeddingModel import init_emb_table
from model.LightGCN import get_lightgcn_out_emb
from data.csr_graph_helper import numba_csr_mult_dense

import numpy as np
import torch
import os.path as osp
import setproctitle
import time


class Propagation_Model:
    
    def __init__(self, config):
        self.config = config
        
        data_root = self.config['data_root']
        info = io.load_yaml(osp.join(data_root, 'info.yaml'))
        self.num_nodes = info['num_nodes']
        
        config['device'] = 'cpu'
        config['freeze_emb'] = 1  # true
        self.emb_table = init_emb_table(config, self.num_nodes).weight
        
        self.prop_type = self.config['prop_type']
        if self.prop_type == 'pprgo':
            print("## load ppr neighbors and ppr weights ...")
            raw_nei = io.load_pickle(osp.join(config['ppr_data_root'], "nei.pkl"))
            raw_wei = io.load_pickle(osp.join(config['ppr_data_root'], "wei.pkl"))
            
            topk = config['topk']
            self.nei = torch.LongTensor(raw_nei[:, : topk])
            self.wei = torch.FloatTensor(raw_wei[:, : topk])
            
            if config['use_uniform_weight']:
                print("## uniform weight")
                _w = torch.ones(self.nei.shape)
                _w[self.wei == 0] = 0
                self.wei = _w / (_w.sum(dim=-1, keepdim=True) + 1e-12)
            else:
                print("## not uniform weight")
                self.wei = self.wei / (self.wei.sum(dim=-1, keepdim=True) + 1e-12)
            
        else:
            assert self.prop_type == 'lightgcn'
            E_src = io.load_pickle(osp.join(data_root, 'train_undi_csr_src_indices.pkl'))
            E_dst = io.load_pickle(osp.join(data_root, 'train_undi_csr_indices.pkl'))
            indptr = io.load_pickle(osp.join(data_root, 'train_undi_csr_indptr.pkl'))
            all_degrees = indptr[1:] - indptr[:-1]
            d_src = all_degrees[E_src]
            d_dst = all_degrees[E_dst]
            
            edge_weights = torch.FloatTensor(1 / (d_src * d_dst)).sqrt()
            del all_degrees, d_src, d_dst
            
            if self.config['use_numba_csr_mult']:
                self.indptr = indptr
                self.indices = E_dst
                self.edge_weights = edge_weights.numpy()
            elif hasattr(torch, 'sparse_csr_tensor'):
                self.A = torch.sparse_csr_tensor(
                    torch.LongTensor(np.array(indptr, dtype=np.int64)),
                    torch.LongTensor(E_dst),
                    edge_weights,
                    (self.num_nodes, self.num_nodes)
                )
            else:
                del indptr
                E_src = torch.IntTensor(E_src)
                E_dst = torch.IntTensor(E_dst)
                E = torch.cat([E_src, E_dst]).reshape(2, -1)
                del E_src, E_dst
                self.A = torch.sparse_coo_tensor(
                    E, edge_weights, 
                    (self.num_nodes, self.num_nodes)
                )
    
    def do_propagation(self):
        for _ in range(self.config['prop_times']):
            self._do_propagation()
    
    def _do_propagation(self):
        print("## do_propagation:", self.prop_type)
        if self.prop_type == 'pprgo':
            _emb_table = torch.empty(self.emb_table.shape, dtype=torch.float32)
            dl = torch.utils.data.DataLoader(dataset=torch.arange(len(self.emb_table)),
                                             batch_size=8192)
            for nids in dl:
                _emb_table[nids] = self._calc_pprgo_out_emb(nids)
            
            self.emb_table = _emb_table
            
        else:  # self.prop_type == 'lightgcn'
            if self.config['use_numba_csr_mult']:
                print("- use_numba_csr_mult, do not stack")
                X_in = self.emb_table.numpy()
                X_out = np.empty(X_in.shape, dtype=np.float32)
                numba_csr_mult_dense(
                    self.indptr, self.indices, self.edge_weights,
                    X_in, X_out
                )
                self.emb_table = torch.FloatTensor(X_out)
                del X_in
            else:
                self.emb_table = get_lightgcn_out_emb(
                    self.A, self.emb_table, self.config['num_gcn_layers'],
                    stack_layers=self.config['stack_layers']
                )
        self.emb_table /= self.emb_table.abs().mean()
        
    def _calc_pprgo_out_emb(self, nids):
        top_nids = self.nei[nids]
        top_weights = self.wei[nids]
        
        top_embs = self.emb_table[top_nids]
        top_weights = top_weights
        
        out_embs = torch.matmul(top_weights.unsqueeze(-2), top_embs)
        return out_embs.squeeze()


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    model = Propagation_Model(config)
    model.do_propagation()
    
    torch.save(model.emb_table, osp.join(results_root, 'out_emb_table.pt'))
    print("## done!")


if __name__ == "__main__":
    
    setproctitle.setproctitle('propagation-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
