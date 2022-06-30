import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict
from model.BaseEmbeddingModel import BaseEmbeddingModel
from helper.build_val_test_dl import build_eval_dl

import numpy as np
import torch
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    data_root = config['data_root']
    
    data = {}
    model = BaseEmbeddingModel(config, data)
    model.load_file(config['file_input'])
    
    data['model'] = model
    
    info = io.load_yaml(osp.join(data_root, 'info.yaml'))
    
    all_nids = np.arange(info['num_nodes'])
    np.random.seed(2022)
    np.random.shuffle(all_nids)
    case_study_nids = all_nids[:config['num_sample']]
    dl = torch.utils.data.DataLoader(case_study_nids, batch_size=128)
    
    topk = config['topk']
    R = np.empty(shape=(len(case_study_nids), topk), dtype=np.int64)
    st = 0
    for src in tqdm(dl):
        src = src.numpy()
        all_target_score = model.infer_whole_graph(src, mask_nei=True)
        
        _, batch_R = torch.tensor(all_target_score).topk(topk)
        
        R[st : st + len(batch_R)] = batch_R.numpy()
        st += len(batch_R)
    
    io.save_pickle(config['file_output'], R)
    
    # eval_dl = build_eval_dl(
    #     eval_method=config['test_method'],
    #     file_eval=config['file_test'],
    #     mask_nei=config['mask_nei_when_test'],
    #     batch_size=128,
    #     data=data
    # )
    
    # topk = config['topk']
    # R = np.empty(shape=(eval_dl.num_samples(), topk), dtype=np.int64)
    # st = 0
    # for batch_data in tqdm(eval_dl, desc='infer'):
    #     all_target_score = model.eval_a_batch(batch_data, only_return_all_target_score=True)
        
    #     _, batch_R = torch.tensor(all_target_score).topk(topk)
        
    #     R[st : st + len(batch_R)] = batch_R.numpy()
    #     st += len(batch_R)
    
    # TODO: for one_pos_k_neg cases, map node id in R
    
    # io.save_pickle(config['file_output'], R)


if __name__ == "__main__":
    
    setproctitle.setproctitle('get_emb_top_reco-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
