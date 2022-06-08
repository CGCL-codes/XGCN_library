import sys
# PROJECT_ROOT = sys.argv[1]
PROJECT_ROOT='/home/sxr/code/xgcn'
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import ensure_dir

import numpy as np
import torch
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt
import os.path as osp
import setproctitle
import time


def plot_density(ax, data, x, smoothing=0.5, color='b', label=''):
    density = gaussian_kde(data)
    density.covariance_factor = lambda : smoothing
    density._compute_covariance()
    ax.plot(x, density(x), color=color, label=label)


def get_norm(file):
    E = torch.load(file, map_location='cpu').detach()
    return E.norm(dim=-1).numpy()


def main():
    
    dataset_name = 'Pokec'
    # the .svg figures will be saved at results_root
    results_root = '/home/sxr/data/social_and_user_item/_model_outputs/emb_stat'
    ensure_dir(results_root)

    print("plot top reco indegree")
    d_n2v = io.load_pickle('/home/sxr/data/social_and_user_item/_model_outputs/gnn_livejournal/livejournal_node2vec_top100_reco.pkl')
    d_pprgo = io.load_pickle('/home/sxr/data/social_and_user_item/_model_outputs/gnn_livejournal/plot/pprgo/[best][bpr][reg0]/top_reco_indegrees.pkl')
    d_xgcn = io.load_pickle('/home/sxr/data/social_and_user_item/_model_outputs/gnn_livejournal/plot/dnn-lightgcn-final_version/[gcn2layer][scale][3layer][freq1][K10][endure3]/top_reco_indegrees.pkl')

    d_n2v = d_n2v.flatten()
    d_pprgo = d_pprgo.flatten()
    d_xgcn = d_xgcn.flatten()
    
    x = np.arange(0, 800)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)

    plot_density(ax, d_n2v, x, smoothing=0.05, color='g', label='node2vec')
    plot_density(ax, d_pprgo, x, smoothing=0.05, color='b', label='pprgo')
    plot_density(ax, d_xgcn, x, smoothing=0.05, color='r', label='xgcn')
    ax.set_title(dataset_name + ' - top100 reco')
    ax.set_xlabel('in-degree')
    ax.set_ylabel('density')
    plt.legend()
    
    fig.savefig(osp.join(results_root, 'top100_reco_indegree.svg'), format='svg', dpi=200)
    # plt.show()
    
    print("plot embedding norm")
    norm_n2v = get_norm('/home/sxr/data/social_and_user_item/_model_outputs/gnn_pokec/n2v_best/out_emb_table.pt')
    norm_pprgo = get_norm('/home/sxr/data/social_and_user_item/_model_outputs/gnn_pokec/plot/pprgo/[best][bpr][reg0]/out_emb_table.pt')
    norm_xgcn = get_norm('/home/sxr/data/social_and_user_item/_model_outputs/gnn_pokec/plot/dnn-lightgcn-final_version/[gcn1layer][scale][3layer][freq3][K10][endure3]/out_emb_table.pt')
    
    x = np.arange(0, 10, 0.05)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)

    plot_density(ax, norm_n2v, x, smoothing=0.05, color='g', label='node2vec')
    plot_density(ax, norm_pprgo, x, smoothing=0.05, color='b', label='pprgo')
    plot_density(ax, norm_xgcn, x, smoothing=0.05, color='r', label='xgcn')
    ax.set_title(dataset_name + ' - embedding L2 norm')
    ax.set_xlabel('L2 norm')
    ax.set_ylabel('density')
    plt.legend()
    
    fig.savefig(osp.join(results_root, 'emb_L2_norm.svg'), format='svg', dpi=200)
    # plt.show()
    

if __name__ == "__main__":
    
    setproctitle.setproctitle('plot_density-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
