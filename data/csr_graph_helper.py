import numpy as np
import numba
import scipy.sparse as spsp
import torch


@numba.jit(nopython=True)
def get_neighbors(indptr, indices, u):
    return indices[indptr[u] : indptr[u + 1]]


def from_scipy_to_torch(Asp: spsp.csr_matrix):
    crow_indices = Asp.indptr
    col_indices = Asp.indices
    values = Asp.data
    size = Asp.shape
    A = torch.sparse_csr_tensor(
        torch.tensor(crow_indices),
        torch.tensor(col_indices),
        torch.tensor(values),
        size=size
    )
    return A


@numba.jit(nopython=True, parallel=True)
def numba_csr_mult_dense(indptr, indices, data, X_in, X_out):
    for u in numba.prange(len(indptr) - 1):
        start = indptr[u]
        end = indptr[u + 1]
        # if start == end:
        #     X_out[u] = X_in[u]
        # else:
        u_nei = indices[start : end]
        u_nei_data = data[start : end].reshape(-1, 1)
        X_out[u] = (u_nei_data * X_in[u_nei]).sum(axis=-2)


@numba.jit(nopython=True, parallel=True)
def get_src_indices(indptr):
    num_nodes = len(indptr) - 1
    num_edges = indptr[-1]
    src_indices = np.empty(num_edges, dtype=np.int32)
    for u in numba.prange(num_nodes):
        src_indices[indptr[u] : indptr[u + 1]] = u
    return src_indices


@numba.jit(nopython=True)
def from_edges_to_csr(E_src, E_dst, num_nodes):
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
    offset = np.zeros(num_nodes, dtype=np.int32)
    for i in range(num_edges):
        src = E_src[i]
        dst = E_dst[i]
        start = indptr[src]
        indices[start + offset[src]] = dst
        offset[src] += 1
    
    return indptr, indices


@numba.jit(nopython=True)
def remove_repeated_edges_in_csr(indptr, indices):
    '''
        Note: content in indices will be changed
    '''
    num_nodes = len(indptr) - 1
    new_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    start = np.int64(0)
    for u in range(num_nodes):
        nei = indices[indptr[u] : indptr[u + 1]]
        unique_nei = np.array(list(set(list(nei))), dtype=np.int32)
        
        new_indptr[u] = start
        new_degree = len(unique_nei)
        indices[start : start + new_degree] = unique_nei
        start += new_degree
    
    new_indptr[-1] = start
    new_indices = indices[:start]
    
    return new_indptr, new_indices


@numba.jit(nopython=True)
def remove_edges_in_csr(indptr, indices, rm_edge_mask):
    '''
        Note: 
            * rm_edge_mask should correspond to indices
            * content in indices will be changed
    '''
    indices[rm_edge_mask] = -1  # mask removed edges
    
    num_nodes = len(indptr) - 1
    new_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    start = np.int64(0)
    for u in range(num_nodes):
        new_indptr[u] = start
        nei = indices[indptr[u] : indptr[u + 1]]
        du = 0  # new_degree of u
        for v in nei:
            if v != -1:
                indices[start + du] = v
                du += 1
        start += du
    
    new_indptr[-1] = start
    new_indices = indices[:start]
    
    return new_indptr, new_indices
