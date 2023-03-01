Prepare the Validation/Test Sets
===================================

.. code:: python

    >>> 
    >>> num_validation = 2000
    >>> val_edges = pos_edges[:num_validation]     # edges for validation
    >>> test_edges = pos_edges[num_validation:]    # edges for test
    >>> 
    >>> val_set = XGCN.data.from_edges_to_adj_eval_set(val_edges)    # convert the edges to adjacency list
    >>> test_set = XGCN.data.from_edges_to_adj_eval_set(test_edges)
    >>> io.save_pickle(osp.join(data_root, 'val_set.pkl'), val_set)
    >>> io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)

Your data directory will be like:

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_facebook
        |   ├── raw_graph.txt
        |   └── csr
        └── instance_facebook
            ├── info.yaml
            ├── indices.pkl
            ├── indptr.pkl
            ├── pos_edges.pkl
            ├── test_set.pkl
            └── val_set.pkl
