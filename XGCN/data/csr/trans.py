import dgl

def to_dgl(indptr, indices):
    g = dgl.graph(('csr', (indptr, indices, [])))
    return g
