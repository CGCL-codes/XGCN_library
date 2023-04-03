Dataset Instance
=======================

In XGCN, before running the models, datasets must be processed into a standard format. 
We call such processed data as "dataset instance", 
which is basically a directory containing several formatted data files. 
For example, a dataset instance of the facebook dataset may look like follows: 

.. code::

    instance_facebook
    ├── info.yaml      # contains some basic information such as "graph type" and "number of nodes"
    ├── indptr.pkl     # graph for training in CSR format
    ├── indices.pkl
    ├── val_set.pkl    # evaluation sets
    └── test_set.pkl

XGCN has no restriction on the name of the dataset instance directory, 
but we recommend to name it as ``instance_[dataset]`` for clarity. 

We also recommend to arrange all the data (datasets and model outputs) with a clear directory structure. 
For example, from the beginning, you may manually setup a ``XGCN_data`` directory as follows: 

.. code:: 

    XGCN_data
    └── dataset
        └── raw_facebook
            └── facebook_combined.txt

After some further data processing and model running, your directory may look like: 

.. code:: 

    XGCN_data
    ├── dataset
    |   ├── raw_facebook         # raw data
    |   ├── instance_facebook    # dataset instance
    |   ├── raw_xxx
    |   ├── instance_xxx
    └── model_output
        └─facebook
          ├── GraphSAEG    # saved model and evaluation results
          └── xGCN

You can find some detailed data processing and model running examples in the 
Running Examples section. In our examples, we'll follow the above directory structure. 

info.yaml
------------------

``info.yaml`` contains some basic information such as "graph type" and "number of nodes". 
For homogenous graphs (e.g. social networks), the contents should include:

.. code:: yaml

    graph_type: homo
    num_nodes: [int value]
    num_edges: [int value]

For bipartite graphs (e.g. user-item graphs), the contents should include:

.. code:: yaml

    graph_type: user-item
    num_nodes: [int value]
    num_edges: [int value]
    num_users: [int value]
    num_items: [int value]

You can save/load .yaml files using ``XGCN.data.io``:

.. code:: python

    from XGCN.data import io

    info = io.load_yaml('info.yaml')  # load
    io.save_yaml('info.yaml', info)   # save

CSR Graph
------------------------------

``indptr.pkl`` and ``indices.pkl`` is the graph for training in CSR format. 
They are numpy arrays saved using ``pickle``. You can save/load objects with ``pickle`` 
by using ``XGCN.data.io``: 

.. code:: python

    indptr = io.load_pickle('indptr.pkl')  # load
    io.save_pickle('indptr.pkl', indptr)   # save

`CSR <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ 
is a compact format for sparse metrices. XGCN use this structure to save 
graphs' adjacency matrices and implements some algorithoms. The reasons 
for us to use this format are:

* (1) High-efficency. CSR format is efficient on some key graph/matrix operations such as "querying node neighbors" (O(1) time complexity). By using `Numba <https://numba.pydata.org/>`_ for acceleration based on the CSR data structure, XGCN provides some efficient implements such as random walk, PPR (Personalized PageRank), and ItemCF. 

* (2) Memory-saving. The existing open-source packages for sparse matrix multiplication tend to use too much memory. Though slower than PyTorch's implementation, XGCN implements a Numba-based CSR-matrix-with-dense-matrix multiplication, which consumes lesser memory. **(To add some experiments here)**

* (3) Friendly with DGL's API. DGLGraph can be initialized directly from the CSR format.

Evaluation Sets
---------------------

In link prediction tasks, A single evaluation sample can be formulated as: 
(src, pos[1], ..., pos[m], neg[1], ... neg[k]), where src, pos, neg denotes source node, 
positive node, and negative node, respectively. 
The positive nodes usually comes from the removed edges from the original graph. 
The negative nodes are usually sampled from un-interacted nodes 
(i.e. nodes that are not neighbors of the source node). 

Considering the number of positive nodes and negative nodes for each source node, 
XGCN supports three kinds of evaluation methods: 

* "one-pos-k-neg"

* "whole-graph-one-pos"

* "whole-graph-multi-pos"

For "one-pos-k-neg", each evaluation sample has one positive node and k negative nodes. 
Different evaluation samples may have the same source node. 
The saved pickle file should be a N*(2+k) numpy array, for example: 

.. code:: 

    X = np.array([
        [0, 1, 33, 102, 56, ... ], 
        [0, 2, 150, 98, 72, ... ], 
        [2, 4, 203, 42, 11, ... ],
        [5, 0, 64, 130, 10, ... ],
        ...
    ])

The first column is the source nodes, the second column is the positive nodes, 
and the rest is the negative nodes. 

For "one-pos-whole-graph", each evaluation sample has one positive node. 
Different evaluation samples may have the same source node. 
We consider all the un-interacted nodes in the graph as negative samples. 
The saved pickle file should be a N*2 numpy array, for example: 

.. code:: python

    X = np.array([
        [0, 1], 
        [0, 2], 
        [2, 4],
        [5, 0],
        ...
    ])

For "multi-pos-whole-graph", we also consider all the un-interacted nodes as negative samples. 
Each evaluation sample has one or more positive nodes. 
Different evaluation samples should have different source nodes. 
The saved object should be a Dict like follows: 

.. code:: python

    eval_set = {
        'src': np.array([0, 2, 5, ... ]),
        'pos_list': [
            np.array([1, 2]), 
            np.array([4, ]), 
            np.array([0, ]), 
            ...
        ]
    }

The 'src' field of the Dict is a numpy array of the source nodes. 
The 'pos_list' field of the Dict is a list of numpy array of the positive nodes. 

We don't restrict filenames for the evaluation sets. 
The evaluation method and the corresponding file can be specified in the model configuration.
