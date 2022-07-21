import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from utils import io

import numpy as np
import pandas as pd
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def get_word_emb(token, dict_token2id, word_emb_table):
    if token in dict_token2id:
        return 0, word_emb_table[dict_token2id[token]]
    else:  # token not found
        # emb = np.random.normal(
        #     loc=0.0, scale=word_emb_table_std, size=(word_emb_table.shape[-1],))
        emb = np.zeros(word_emb_table.shape[-1])
        return 1, emb


def process_token_feat(token_list, dict_token2id, word_emb_table):
    N = len(token_list)
    X = np.empty((N, word_emb_table.shape[-1]),
                 dtype=np.float32)
    cnt_token_not_found = 0
    for i, token in tqdm(enumerate(token_list), total=N):
        inc, emb = get_word_emb(token, dict_token2id, word_emb_table)
        cnt_token_not_found += inc
        X[i] = emb
    print("# total tokens:", len(token_list))
    print("# token not found cases:", cnt_token_not_found)
    return X


def process_token_seq_feat(token_seq_list, dict_token2id, word_emb_table, merge='mean'):
    cnt_token = 0
    cnt_token_not_found = 0
    N = len(token_seq_list)
    X = np.empty((N, word_emb_table.shape[-1]),
                 dtype=np.float32)
    for i, token_seq in tqdm(enumerate(token_seq_list), total=N):
        emb_list = []
        token_seq = token_seq.split()
        for token in token_seq:
            inc, emb = get_word_emb(token.lower(), dict_token2id, word_emb_table)
            cnt_token += 1
            cnt_token_not_found += inc
            emb_list.append(emb)
        
        if merge == 'sum':
            emb = np.sum(emb_list, axis=0)
        elif merge == 'mean':
            emb = np.mean(emb_list, axis=0)
        else:
            assert 0
        
        X[i] = emb
    print("# total tokens:", cnt_token)
    print("# token not found cases:", cnt_token_not_found)
    return X


def main():
    
    # process RecBole user/item features
    
    config = parse_arguments()
    print_dict(config)
    
    file_input = config['file_input']  # .user/.item file
    data_root = config['data_root']  # word embedding data root
    results_root = config['results_root']  # place to save processed feat
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    df = pd.read_csv(file_input, delimiter='\t', header=0)
    
    dict_token2id = io.load_pickle(osp.join(data_root, 'dict_token2id.pkl'))
    word_emb_table = io.load_pickle(osp.join(data_root, 'word_embeddings.pkl'))

    for column_name in df.columns:
        feat_name, feat_type = column_name.split(':')
        if feat_name not in config['feat_to_process']:
            continue
        else:
            print("# process feat: [{}], type: [{}]".format(feat_name, feat_type))
            df[column_name].fillna('nan', inplace=True)  # fill Nan as 'nan'
            if feat_type == 'token':
                X = process_token_feat(df[column_name], 
                                       dict_token2id, word_emb_table)
            elif feat_type == 'token_seq':
                X = process_token_seq_feat(df[column_name], 
                                           dict_token2id, word_emb_table, merge='mean')
            io.save_pickle(osp.join(results_root, 
                                    feat_name + '-' + feat_type + '.pkl'), X)
    

if __name__ == '__main__':
    
    setproctitle.setproctitle('process_token_feat-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
