import numpy as np

from numba import njit


@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    if indptr[t] < indptr[t + 1]:
        return indices_or_data[indptr[t] : indptr[t + 1]]
    return np.array([t], dtype=np.int32)


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    walk[1] = np.random.choice(_neighbors(indptr, indices, t))
    for j in range(2, walk_length):
        ##--
        if indptr[walk[j - 1]] >= indptr[walk[j - 1] + 1]:
            break
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = np.random.choice(neighbors)
            continue
        while True:
            new_node = np.random.choice(neighbors)
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, t):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = np.empty(walk_length, dtype=indices.dtype)
    walk[0] = t
    walk[1] = _neighbors(indptr, indices, t)[
        np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
    ]
    for j in range(2, walk_length):
        neighbors = _neighbors(indptr, indices, walk[j - 1])
        neighbors_p = _neighbors(indptr, data, walk[j - 1])
        if p == q == 1:
            # faster version
            walk[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            continue
        while True:
            new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
            r = np.random.rand()
            if new_node == walk[j - 2]:
                if r < prob_0:
                    break
            elif _isin_sorted(_neighbors(indptr, indices, walk[j - 2]), new_node):
                if r < prob_1:
                    break
            elif r < prob_2:
                break
        walk[j] = new_node
    return walk


class RandomWalkGraph:
    
    def __init__(self, indptr, indices):
        self.indptr = indptr
        self.indices = indices
        self.num_nodes = len(indptr) - 1

    def generate_random_walk(self, walk_length, p, q, start):
        walk = _random_walk(self.indptr, self.indices, walk_length, p, q, start)
        return walk
