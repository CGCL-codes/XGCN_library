import numba


@numba.jit(nopython=True, parallel=True)
def csr_mult_dense(indptr, indices, data, X_in, X_out):
    for u in numba.prange(len(indptr) - 1):
        start = indptr[u]
        end = indptr[u + 1]
        # if start == end:
        #     X_out[u] = X_in[u]
        # else:
        u_nei = indices[start : end]
        u_nei_data = data[start : end].reshape(-1, 1)
        X_out[u] = (u_nei_data * X_in[u_nei]).sum(axis=-2)
