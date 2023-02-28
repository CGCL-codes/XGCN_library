import os.path as osp
import subprocess
import pathlib
import numpy as np
import torch
    

def wc_count(file_name):
    ## count file's lines
    assert osp.exists(file_name)
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def print_dict(d):
    for key in d:
        print(key, ":", d[key])


def get_formatted_results(r):
    s = ""
    for key in r.keys():
        s += "{}:{:.4f} || ".format(key, r[key])
    return s


def combine_dict_list_and_calc_mean(dict_list, weights=None):
    d = {}
    if weights is not None:
        for key in dict_list[0]:
            d[key] = np.array([weights[i] * dict_list[i][key] for i in range(len(dict_list))]).sum()
    else:
        for key in dict_list[0]:
            d[key] = np.array([dict_list[i][key] for i in range(len(dict_list))]).mean()
    return d


def set_random_seed(seed=2023):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_idx_mapping(x: torch.tensor):
    idx_mapping = {}
    for i, a in enumerate(x):
        idx_mapping[a.item()] = i
    return idx_mapping


def get_unique(x: torch.tensor):
    unique_x = torch.tensor(
        list(set(x.flatten().tolist())), dtype=x.dtype
    )
    idx_mapping = get_idx_mapping(unique_x)
    return unique_x, idx_mapping


def id_map(X: torch.tensor, mapping_dict: dict):
    X_flatten = X.flatten()
    Y_flatten = torch.empty((len(X_flatten),), dtype=X.dtype)
    for i, x in enumerate(X_flatten):
        Y_flatten[i] = mapping_dict[x.item()]
    Y = Y_flatten.reshape(X.shape)
    return Y
