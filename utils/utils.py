import pathlib
import os.path as osp
import numpy as np
import numba
import torch
import dgl


class ReIndexDict:
    
    def __init__(self):
        self.d = {}
        self.rev_d = None
        self.cnt = 0
        
    def __getitem__(self, old_id):
        if old_id in self.d:
            return self.d[old_id]
        else:
            new_id = self.cnt
            self.d[old_id] = new_id
            self.cnt += 1
            return new_id
        
    def get_old2new_dict(self):
        return self.d
    
    def get_new2old_dict(self):
        if self.rev_d is None:
            self.rev_d = {}
            for old_id in self.d:
                self.rev_d[self.d[old_id]] = old_id
        return self.rev_d


def set_random_seed(seed=2022):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def element_wise_map(fn, t, dtype=None):
    
    if dtype is None:
        dtype = t.dtype
    new_t = torch.empty(size=t.size(), dtype=dtype)

    _t = t.view(-1).cpu().numpy()
    _new_t = new_t.view(-1).cpu().numpy()

    for i in range(len(_t)):
        _new_t[i] = fn(_t[i])

    return new_t


def wc_count(file_name):
    ## count file's lines
    assert osp.exists(file_name)
    import subprocess
    out = subprocess.getoutput("wc -l %s" % file_name)
    return int(out.split()[0])


def gram_schmidt(U):
    
    def normalize(X):
        return X / torch.norm(X, p=2)
    
    V = torch.FloatTensor(size=U.shape)
    V[0] = normalize(U[0])
    for i in range(1, len(U)):
        ui = U[i]
        vs = V[:i]
        V[i] = normalize(ui - (ui @ vs.T) @ vs)
    return V


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def print_dict(d):
    for key in d:
        print(key, ":", d[key])


@numba.jit(nopython=True)
def find_first(item, vec):
    '''return the index of the first occurence of item in vec'''
    for i in range(len(vec)):
        if item == vec[i]:
            return i
    return -1


def combine_dict_list_and_calc_mean(dict_list, weights=None):
    d = {}
    if weights is not None:
        for key in dict_list[0]:
            d[key] = np.array([weights[i] * dict_list[i][key] for i in range(len(dict_list))]).sum()
    else:
        for key in dict_list[0]:
            d[key] = np.array([dict_list[i][key] for i in range(len(dict_list))]).mean()
    return d


def get_formatted_results(r):
    s = ""
    for key in r.keys():
        s += "{}:{:.4f} || ".format(key, r[key])
    return s


def get_ui_degree_and_id(g):
    user_flag = g.ndata['user_flag']
    
    all_id = g.nodes()
    out_d = g.out_degrees()
    in_d = g.in_degrees()
    
    mask = user_flag == 1
    user_id = all_id[mask].numpy()
    item_id = all_id[~mask].numpy()
    
    user_degree = out_d[mask].numpy()
    item_degree = in_d[~mask].numpy()
    
    num_users = len(user_id)
    num_items = len(item_id)
    # print('num_user:', num_users)
    # print('num_item:', num_items)
    
    return user_id, item_id, user_degree, item_degree, num_users, num_items
