import dgl


def to_dgl(indptr, indices):
    return dgl.graph(('csr', (indptr, indices, [])))
