import numba


@numba.jit(nopython=True)
def get_neighbors(indptr, indices, u):
    return indices[indptr[u] : indptr[u + 1]]


@numba.jit(nopython=True)
def get_degrees(indptr):
    return indptr[1:] - indptr[:-1]
