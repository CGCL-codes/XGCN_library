Dara Processing Example: facebook
=====================================

Let's begin with a small social network dataset: facebook.
The data is included in our XGCN repository: 
``data/raw_facebook/``. You can also download it from SNAP: 
`facebook_combined.txt.gz <http://snap.stanford.edu/data/facebook_combined.txt.gz>`_. 

Here assume that we don't have existing evaluation set 
and want to split some edges for model evaluation.

We recommend to arrange the data with a clear directory structure. 
To get started, you may manually setup a ``XGCN_data`` directory as follows: 
(It's recommended to put your ``XGCN_data`` somewhere else than in this repository.)

.. code:: 

    XGCN_data
    └── dataset
        └── raw_facebook
            └── facebook_combined.txt

If you like to do the data process in JupyterNotebook, please refer to 
``script/data_process/facebook/dataset_instance_making.ipynb``.

Firstly, we import some modules: 

.. code:: python

    >>> import dgl
    >>> import XGCN
    >>> from XGCN.data import io
    >>> from XGCN.utils.utils import ensure_dir, set_random_seed
    >>> import os.path as osp

Specify the data root and the dataset name: 

.. code:: python

    >>> all_data_root = '../../XGCN_data'  # write your own data root here
    >>> dataset = 'facebook'

Load the ``facebook_combined.txt`` as numpy array, and convert the edge-list to DGLGraph: 

.. code:: python

    >>> raw_data_root = osp.join(all_data_root, 'dataset/raw_' + dataset)
    >>> file_raw_graph = osp.join(raw_data_root, 'facebook_combined.txt')
    >>> E_src, E_dst = io.load_edges(file_raw_graph)
    >>> print(E_src)
    [   0    0    0 ... 4027 4027 4031]
    >>> print(E_dst)
    [   1    2    3 ... 4032 4038 4038]
    >>> g = dgl.graph((E_src, E_dst))

Loading large graphs from the text data can be time-consuming. 
We can cache the raw graph using ``pickle``: 

.. code:: python

    >>> io.save_pickle(osp.join(raw_data_root, 'raw_g.pkl'), g)

To evaluate a link prediction model, it is common to split a portion of edges as 
positive samples. To do this, you can use the function ``XGCN.data.split_edges.split_edges``: 

.. code:: python

    >>> set_random_seed(1999)
    >>> num_sample = 10_000       # number of edges to split
    >>> min_src_out_degree = 3    # guarantee the minimum out-degree of a source node
    >>> min_dst_in_degree = 1     # guarantee the minimum in-degree of a destination node
    >>> 
    >>> g, pos_edges = XGCN.data.split_edges.split_edges(g, num_sample, min_src_out_degree, min_dst_in_degree)
    # init CSR_Graph_rev_rm_edge...
    sampling edges 9999/10000 (99.99%)
    num sampled edges: 10000
    # csr.to_compact(g.indptr, g.indices)...

Now we have all the positive edges: ``pos_edges``, let's divide them for 
validation set and test set, and we use the "multi-pos-whole-graph" evaluation method:

.. code:: python

    >>> num_validation = 2000
    >>> val_edges = pos_edges[:num_validation]     # edges for validation
    >>> test_edges = pos_edges[num_validation:]    # edges for test
    >>> val_set = XGCN.data.split_edges.from_edges_to_adj_eval_set(val_edges)    # convert the edges to adjacency list
    >>> test_set = XGCN.data.split_edges.from_edges_to_adj_eval_set(test_edges)


At last, to form a complete dataset instance, we need to add a ``info`` Dict: 

.. code:: python

    >>> info = {'graph_type': 'homo', 'num_nodes': g.num_nodes(), 'num_edges': g.num_edges()}

Now we have already generated a complete dataset instance, let's save it:

.. code:: python

    >>> data_root = osp.join(all_data_root, 'dataset/instance_' + dataset)
    >>> ensure_dir(data_root)  # make the directory if it doesn't exist
    >>> io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(data_root, 'g.pkl'), g)
    >>> io.save_pickle(osp.join(data_root, 'pos_edges.pkl'), pos_edges)
    >>> io.save_pickle(osp.join(data_root, 'val_set.pkl'), val_set)
    >>> io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)

Here we also save the ``pos_edges``, so you can use it to make evaluation sets for 
"one-pos-k-neg" or "one-pos-whole-graph" method by concatenating some randomly 
sampled negative nodes. 

If you have done the above steps successfully, your data directory will be like follows: 

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_facebook
        |   ├── facebook_combined.txt
        |   └── raw_g.pkl
        └── instance_facebook
            ├── info.yaml
            ├── g.pkl
            ├── pos_edges.pkl
            ├── test_set.pkl
            └── val_set.pkl
