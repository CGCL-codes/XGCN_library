Data Preparation
======================

GNN_ZOO supports handling both social graphs (all the nodes are users) and user-item graphs. 
In this section, let's take a small dataset - `facebook <http://snap.stanford.edu/data/ego-Facebook.html>`_ - as an example, 
start from the raw ``.txt`` file of a graph and go through the whole data preparation pipeline. 

The facebook data follows the edge-list format and is included in our GNN_ZOO repository: ``example_data/raw_facebook/facebook_combined.txt``. 
You can also download it from SNAP: `facebook_combined.txt.gz <http://snap.stanford.edu/data/facebook_combined.txt.gz>`_. 


1. Prepare the Raw Data
-----------------------------

Firstly, let's make an empty directory named ``gnn_zoo_data`` (or any one you like) to place all the datasets and model outputs. 
It's recommended to put ``gnn_zoo_data`` somewhere else than in this repository. 

GNN_ZOO supports two kinds of text graph data as input: 

(1) **Edge list**. One line represents an edge: (source node, destination node). The nodes are seperated by a whitespace, 
for example::

    0 1
    0 2
    1 4
    2 1
    2 3
    2 4

(2) **Adjacency table**. One line represents a source node and its destination nodes. Two nodes are seperated by a whitespace, 
for example:: 

    0 1 2
    1 4
    2 1 3 4

We recommend to arrange the data with a clear directory structure. 
From the beginning, you may manually setup the ``gnn_zoo_data`` directory as follows, 
where ``raw_graph.txt`` is renamed from ``facebook_combined.txt`` for consistency. 

.. code:: 

    gnn_zoo_data
    └── dataset
        └── raw_facebook
            └── raw_graph.txt

After some further data processing and model running, the directory may look like: 

.. code:: 

    gnn_zoo_data
    ├── dataset
    |   ├── raw_facebook         # raw data
    |   ├── instance_facebook    # processed graph and evaluation sets
    |   ├── raw_xxx
    |   ├── instance_xxx
    └── model_output
        └─facebook
          ├── GraphSAEG    # saved model and evaluation results
          ├── PPR
          └── PPRGo


2. Dataset Instance Making
-----------------------------

Now we have the raw text data, let's move to the dataset instance making! 
Firstly, import some modules: 

.. code:: python

    >>> import gnn_zoo
    >>> from gnn_zoo.utils import io  # to save/load files
    >>> from gnn_zoo.utils.utils import ensure_dir, set_random_seed
    >>> import os.path as osp

Specify the data root and the dataset name: 

.. code:: python

    >>> all_data_root = '../../gnn_zoo_data'  # write your own data root here
    >>> dataset = 'facebook'

Load the ``raw_graph.txt`` as numpy array: 

.. code:: python

    >>> raw_data_root = osp.join(all_data_root, 'dataset/raw_' + dataset)
    >>> file_raw_graph = osp.join(raw_data_root, 'raw_graph.txt')
    >>> E_src, E_dst = io.txt_graph.load_edges(file_raw_graph)
    >>> print(E_src)
    [   0    0    0 ... 4027 4027 4031]
    >>> print(E_dst)
    [   1    2    3 ... 4032 4038 4038]

Convert the edge list to CSR format: 

.. code:: python
    
    >>> info, indptr, indices = gnn_zoo.data.from_edges_to_csr( \
    ...     E_src, E_dst, graph_type='homo')
    # from_edges_to_csr ...
    # remove_repeated_edges ...
    ## 0 edges are removed
    >>> print(info)
    {'graph_type': 'homo', 'num_nodes': 4039, 'num_edges': 88234}

In function ``gnn_zoo.data.from_edges_to_csr``, the argument ``graph_type`` can be 
'homo' (homogeneous) or 'user-item'. For social graphs, fill 'homo'. 
The function returns a ``dict`` containing basic information about the graph and 
the numpy CSR array: ``indptr`` and ``indices``. 

We save the raw graph in CSR format using ``pickle``. And save the graph information 
as ``info.yaml``.

.. code:: python

    >>> raw_csr_root = osp.join(raw_data_root, 'csr')
    >>> ensure_dir(raw_csr_root)
    >>> io.save_yaml(osp.join(raw_csr_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(raw_csr_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(raw_csr_root, 'indices.pkl'), indices)

To evaluate a link prediction model, it is common to split a portion of edges as 
positive samples. To do this, you can use the function ``gnn_zoo.data.edges_split``:

.. code:: python

    >>> set_random_seed(1999)
    >>> num_sample = 10_000       # number of edges to split
    >>> min_src_out_degree = 3    # guarantee the minimum out-degree of a source node
    >>> min_dst_in_degree = 1     # guarantee the minimum in-degree of a destination node
    >>> 
    >>> info, indptr, indices, pos_edges = gnn_zoo.data.edges_split( \
    ...     info, indptr, indices, \
    ...     num_sample, min_src_out_degree, min_dst_in_degree)
    # init CSR_Graph_rev_rm_edge...
    sampling edges 9999/10000 (99.99%)
    num sampled edges: 10000
    # csr.to_compact(g.indptr, g.indices)...
    >>> print(info)  # information of the new graph
    {'graph_type': 'homo', 'num_nodes': 4039, 'num_edges': 78234}
    >>> 
    >>> num_validation = 2000
    >>> val_edges = pos_edges[:num_validation]     # edges for validation
    >>> test_edges = pos_edges[num_validation:]    # edges for test
    >>> 
    >>> val_set = gnn_zoo.data.from_edges_to_adj_eval_set(val_edges)    # convert the edges to adjacency list
    >>> test_set = gnn_zoo.data.from_edges_to_adj_eval_set(test_edges)

Now we have a complete dataset instance: the graph for model training, 
the validation set, and the test set. 
Let's save them together in a new directory:

.. code:: python

    >>> data_root = osp.join(all_data_root, 'dataset/instance_' + dataset)
    >>> ensure_dir(data_root)
    >>> io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(data_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(data_root, 'indices.pkl'), indices)
    >>> io.save_pickle(osp.join(data_root, 'pos_edges.pkl'), pos_edges)
    >>> io.save_pickle(osp.join(data_root, 'val_set.pkl'), val_set)
    >>> io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)

Your data directory will be like:

.. code:: 

    gnn_zoo_data
    └── dataset
        ├── raw_facebook
        |   ├── raw_graph.txt
        |   └── csr
        └── instance_facebook
            ├── indices.pkl
            ├── indptr.pkl
            ├── info.yaml
            ├── pos_edges.pkl
            ├── test_set.pkl
            └── val_set.pkl
