Dataset Instance
=======================

In XGCN, datasets must be processed into a standard format before the model running. 
We call such processed data as "Dataset Instance", 
which is basically a directory containing several formatted data files. 
For example, a Dataset Instance of facebook may look like follows:

.. code::

    instance_facebook
    ├── info.yaml      # some basic information such as "graph type" and "number of nodes"
    ├── indices.pkl    # CSR format (numpy array) of the graph for training
    ├── indptr.pkl
    ├── test_set.pkl   # evaluation set (optional)
    └── val_set.pkl

XGCN has no restriction on the name of the Dataset Instance directory, 
but we recommend to name them as ``instance_[dataset]`` for clarity. 

Only the names of the three files are strictly specified: 
``info.yaml``, ``indices.pkl``, and ``indptr.pkl``. 
With the XGCN APIs, they can easily be generated from the raw ``.txt`` graph data.

``info.yaml`` contains some basic information such as "graph type" and "number of nodes". 
``indices.pkl`` and ``indptr.pkl`` are CSR format of the graph for training, 
They are numpy arrays saved using pickle. 



.. code:: python

    X = [
        [0, 1], 
        [0, 2], 
        [2, 4],
        [5, 0],
        ...
    ]

XGCN supports three kinds of evaluation methods: 
"one-pos-k-neg", "one-pos-whole-graph", and "multi-pos-whole-graph". 
We don't restrict filenames of evaluation sets. 

For "one-pos-k-neg", each positive node is associated with k negative samples. 
The saved pickle file should be a N*(2+k) numpy array, for example: 

.. code:: 

    [
        [0, 1, 33, 102, 56, ... ], 
        [0, 2, 150, 98, 72, ... ], 
        [2, 4, 203, 42, 11, ... ],
        [5, 0, 64, 130, 10, ... ],
        ...
    ]

The first column is the source nodes, the second column is the positive nodes, 
and the rest is the negative nodes. 

For "one-pos-whole-graph", we consider all the un-interacted nodes 
(i.e. nodes that are not neighbors of the source node) in the graph as negative samples. 
The saved pickle file should be a N*2 numpy array, for example: 
