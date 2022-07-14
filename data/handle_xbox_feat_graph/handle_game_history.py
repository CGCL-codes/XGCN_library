import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, wc_count, ensure_dir, ReIndexDict
from data.handle_train_graph import handle_train_graph

import numpy as np
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def handle_user_game_history(file, user_dict_old2new, root_to_save):
    num_lines = wc_count(file)
    E_src = []  # user
    E_dst = []  # game
    game_dic = ReIndexDict()
    
    print("process txt...")
    with open(file, 'r') as f:
        for _ in tqdm(range(num_lines)):
            line = f.readline().split()
            if line[0] in user_dict_old2new: 
                E_src.append(user_dict_old2new[line[0]])
                E_dst.append(game_dic[line[1]])
    E_src = np.array(E_src, dtype=np.int32)
    E_dst = np.array(E_dst, dtype=np.int32)
    
    io.save_pickle(osp.join(root_to_save, 'dict_game_old2new.pkl'), game_dic.get_old2new_dict())
    io.save_pickle(osp.join(root_to_save, 'dict_game_new2old.pkl'), game_dic.get_new2old_dict())
    del game_dic
    
    handle_train_graph(E_src, E_dst, root_to_save, 'game', 'user-item')


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    data_root = config['data_root']
    game_feat_root = osp.join(data_root, 'user_game_graph')
    ensure_dir(game_feat_root)
    file_input = config['file_input']  # user's game history
    
    user_dict_old2new = io.load_pickle(osp.join(data_root, 'dict_old2new.pkl'))
    
    handle_user_game_history(file_input, user_dict_old2new, game_feat_root)
    

if __name__ == '__main__':
    
    setproctitle.setproctitle('handle_game_history-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
