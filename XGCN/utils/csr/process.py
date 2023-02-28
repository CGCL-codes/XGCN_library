import numpy as np
import numba

Nan = -1

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
def get_src_indices(indptr):
    num_nodes = len(indptr) - 1
    num_edges = indptr[-1]
    src_indices = np.empty(num_edges, dtype=np.int32)
    for u in range(num_nodes):
        src_indices[indptr[u] : indptr[u + 1]] = u
    return src_indices


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


def get_reversed(indptr, indices):
    num_nodes = len(indptr) - 1
    src_indices = get_src_indices(indptr)
    rev_indptr, rev_indices = from_edges_to_csr(
        E_src=indices, E_dst=src_indices, num_nodes=num_nodes
    )
    return rev_indptr, rev_indices


def get_undirected(indptr, indices):
    num_nodes = len(indptr) - 1
    src_indices = get_src_indices(indptr)
    undi_E_src = np.concatenate([src_indices, indices])
    undi_E_dst = np.concatenate([indices, src_indices])
    undi_indptr, undi_indices = from_edges_to_csr(undi_E_src, undi_E_dst, num_nodes)
    undi_indptr, undi_indices = remove_repeated_edges_in_csr(undi_indptr, undi_indices)
    
    return undi_indptr, undi_indices


@numba.jit(nopython=True)
def remove_neighbor(indptr, indices, src_nid, nei_nid):
    ptr = indptr[src_nid]
    for offset in range(indptr[src_nid+1] - indptr[src_nid]):
        idx = ptr + offset
        if indices[idx] == nei_nid:
            indices[idx] = Nan
            break


@numba.jit(nopython=True)
def to_compact(indptr, indices):
    '''
        Note: content in indices will be changed
    '''
    num_nodes = len(indptr) - 1
    new_indptr = np.empty(num_nodes + 1, dtype=np.int64)
    start = np.int64(0)
    for u in range(num_nodes):
        x = indices[indptr[u] : indptr[u+1]]
        nei = x[x != Nan]
        
        new_indptr[u] = start
        new_degree = len(nei)
        indices[start : start + new_degree] = nei
        start += new_degree
    
    new_indptr[-1] = start
    new_indices = indices[:start]
    
    return new_indptr, new_indices
