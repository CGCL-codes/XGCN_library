import numpy as np
import numba
import torch
from tqdm import tqdm


def GBP_propagation(indptr, indices, X: torch.FloatTensor, L, w, r, rmax, nr):
    num_nodes = len(indptr) - 1
    print("# get_random_walk_edges")
    S_edges = get_random_walk_edges(indptr, indices, L, nr)
    
    print("# from_edges_to_csr")
    S = []
    for E_src, E_dst, data in S_edges:
        s_indptr, s_indices, values = from_edges_to_csr(E_src, E_dst, data, num_nodes)
        S.append(
            torch.sparse_csr_tensor(
                torch.LongTensor(s_indptr), 
                torch.LongTensor(s_indices),
                torch.FloatTensor(values), size=(num_nodes, num_nodes)
            )
        )
    
    print("# get_Q_R")
    Q, R = get_Q_R(indptr, indices, X.numpy(), L, r, rmax)
    Q = [torch.FloatTensor(Q[i]) for i in range(len(Q))]
    R = [torch.FloatTensor(R[i]) for i in range(len(R))]
    
    D = torch.FloatTensor(np.array(indptr[1:] - indptr[:-1], dtype=np.float32))
    Dr = (D**(r)).reshape(-1, 1)
    
    print("# get_P")
    P = torch.zeros(size=X.shape, dtype=torch.float32)
    for l in range(L + 1):
        sum_SR = torch.zeros(size=X.shape, dtype=torch.float32)
        for t in range(L):
            sum_SR += S[l-t] @ R[t]
        
        P += w[l] * Dr * (Q[l] + sum_SR)
    
    print("# propation done!")
    return P


def get_random_walk_edges(indptr, indices, L, nr):
    num_nodes = len(indptr)  - 1
    
    S_edges = [
        [[], [], []]  # E_src, E_dst, data
        for _ in range(L + 1)
    ]
    '''
        (L, batch_size, nr) E_src
        (L, batch_size, nr) E_dst
        (L, batch_size, nr) data
        (L, batch_size, )  num dst
    '''
    all_nids = np.arange(num_nodes)
    start = 0
    end = 0
    batch_size = 512
    for _ in tqdm(range(int(np.ceil(num_nodes / batch_size)))):
        start = end
        end = start + batch_size
        if end > num_nodes:
            end = num_nodes
        batch_nodes = all_nids[start : end]
        
        S_src, S_dst, S_data, S_num_dst = get_batch_S(
            indptr, indices, batch_nodes, L, nr
        )
        
        for l in range(L + 1):
            for i in range(len(batch_nodes)):
                num_dst = S_num_dst[l][i]
                S_edges[l][0].append(S_src[l][i][:num_dst])
                S_edges[l][1].append(S_dst[l][i][:num_dst])
                S_edges[l][2].append(S_data[l][i][:num_dst])
    
    for l in range(L + 1):
        for i in range(3):
            S_edges[l][i] = np.concatenate(S_edges[l][i])
    
    return S_edges


@numba.jit(nopython=True, parallel=True)
def get_batch_S(indptr, indices, batch_nodes, L, nr):
    batch_size = len(batch_nodes)
    S_src = np.empty(shape=(L + 1, batch_size, nr), dtype=np.int64)
    S_dst = np.empty(shape=(L + 1, batch_size, nr), dtype=np.int64)
    S_data = np.empty(shape=(L + 1, batch_size, nr), dtype=np.float32)
    S_num_dst = np.empty(shape=(L + 1, batch_size), dtype=np.int64)

    inc = np.float32(1.0 / nr)
    for i in numba.prange(len(batch_nodes)):
        source = batch_nodes[i]
        S = [  # collect random walk nodes of diffent walk lengths
            numba.typed.Dict.empty(
                key_type=numba.types.int64,
                value_type=numba.types.float32,
            ) for _ in range(L + 1)
        ]
        S[0][source] = np.float32(1.0)  # length zero
        for _ in range(nr):
            u = source
            for l in range(1, L + 1):
                start, end = indptr[u], indptr[u + 1]
                degree = end - start
                if degree == 0:
                    u = source
                    continue
                v = indices[np.random.randint(low=start, high=end)]
                if v in S[l]:
                    S[l][v] += inc
                else:
                    S[l][v] = inc
                u = v
        
        for l in range(L + 1):
            dst = list(S[l].keys())
            num_dst = len(dst)
            S_num_dst[l][i] = num_dst
            S_dst[l][i][:num_dst] = np.array(dst, dtype=np.int64)
            S_src[l][i][:num_dst] = np.full(num_dst, fill_value=source, dtype=np.int64)
            S_data[l][i][:num_dst] = np.array(list(S[l].values()), dtype=np.float32)
    return S_src, S_dst, S_data, S_num_dst


@numba.jit(nopython=True)
def from_edges_to_csr(E_src, E_dst, data, num_nodes):
    num_edges = len(E_src)
    
    # get degrees
    degrees = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_edges):
        u = E_src[i]
        degrees[u] += 1
    
    # get indptr
    ptr = np.int64(0)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    for u in range(num_nodes):
        indptr[u] = ptr
        ptr += degrees[u]
    indptr[-1] = ptr
    
    # get indices
    indices = np.empty(num_edges, dtype=np.int32)
    new_data = np.empty(num_edges, dtype=np.float32)
    offset = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_edges):
        src = E_src[i]
        start = indptr[src]
        
        indices[start + offset[src]] = E_dst[i]
        new_data[start + offset[src]] = data[i]
        
        offset[src] += 1
    
    return indptr, indices, new_data


def get_Q_R(indptr, indices, X, L, r, rmax):
    Q = [
        np.zeros(shape=X.shape, dtype=np.float32) for _ in range(L + 1)
    ]
    
    D = np.array(indptr[1:] - indptr[:-1], dtype=np.float32)
    D_r = (D**(-r)).reshape(-1, 1)
    R = [D_r * X]
    R.extend([
        np.zeros(shape=X.shape, dtype=np.float32) for _ in range(L)
    ])
    
    num_nodes = len(indptr) - 1
    for l in range(L - 1):
        u_start = 0
        u_end = 0
        batch_size = 512
        for _ in tqdm(range(int(np.ceil(num_nodes / batch_size)))):
            u_start = u_end
            u_end = u_start + batch_size
            if u_end > num_nodes:
                u_end = num_nodes
            _calc_Q_R(indptr, indices, D, u_start, u_end, R[l], R[l+1], Q[l], rmax)

    Q[L] = R[L]
    R[L][:] = 0
    
    return Q, R


@numba.jit(nopython=True)
def _calc_Q_R(indptr, indices, D, u_start, u_end, R_l, R_l_plus_1, Q_l, rmax):
    emb_dim = R_l.shape[-1]
    for u in range(u_start, u_end):
        for k in range(emb_dim):
            Rluk = R_l[u][k]
            if np.abs(Rluk) > rmax:
                neighbor_u = indices[indptr[u] : indptr[u+1]]
                for v in neighbor_u:
                    R_l_plus_1[v][k] += Rluk / D[v]
                Q_l[u][k] = Rluk
                R_l[u][k] = 0
