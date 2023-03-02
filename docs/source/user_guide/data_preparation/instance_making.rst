Dataset Intance Making
=========================

Now we have the raw text data, let's start the dataset instance making! 
Here assume that we don't have existing evaluation set 
and want to split some edges for model evaluation. 

Firstly, import some modules: 

.. code:: python

    >>> import XGCN
    >>> from XGCN.utils import io  # to save/load files
    >>> from XGCN.utils.utils import ensure_dir, set_random_seed
    >>> import os.path as osp

Specify the data root and the dataset name: 

.. code:: python

    >>> all_data_root = '../../XGCN_data'  # write your own data root here
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
    
    >>> info, indptr, indices = XGCN.data.from_edges_to_csr( \
    ...     E_src, E_dst, graph_type='homo')
    # from_edges_to_csr ...
    # remove_repeated_edges ...
    ## 0 edges are removed
    >>> print(info)
    {'graph_type': 'homo', 'num_nodes': 4039, 'num_edges': 88234}

In function ``XGCN.data.from_edges_to_csr``, the argument ``graph_type`` can be 
'homo' (homogeneous) or 'user-item'. For social graphs, fill 'homo'. 
The function returns a ``dict`` containing basic information about the graph and 
the numpy CSR array: ``indptr`` and ``indices``. 

Loading large graphs from the text data can be time-consuming. 
We cache the raw graph in numpy CSR format using ``pickle``. 
And save the graph information as ``info.yaml``:

.. code:: python

    >>> raw_csr_root = osp.join(raw_data_root, 'csr')
    >>> ensure_dir(raw_csr_root)
    >>> io.save_yaml(osp.join(raw_csr_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(raw_csr_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(raw_csr_root, 'indices.pkl'), indices)

To evaluate a link prediction model, it is common to split a portion of edges as 
positive samples. To do this, you can use the function ``XGCN.data.edges_split``:

.. code:: python

    >>> set_random_seed(1999)
    >>> num_sample = 10_000       # number of edges to split
    >>> min_src_out_degree = 3    # guarantee the minimum out-degree of a source node
    >>> min_dst_in_degree = 1     # guarantee the minimum in-degree of a destination node
    >>> 
    >>> info, indptr, indices, pos_edges = XGCN.data.edges_split( \
    ...     info, indptr, indices, \
    ...     num_sample, min_src_out_degree, min_dst_in_degree)
    # init CSR_Graph_rev_rm_edge...
    sampling edges 9999/10000 (99.99%)
    num sampled edges: 10000
    # csr.to_compact(g.indptr, g.indices)...
    >>> print(info)  # information of the new graph
    {'graph_type': 'homo', 'num_nodes': 4039, 'num_edges': 78234}

Now we already have a complete Dataset Instance, let's save it:

.. code:: python

    >>> data_root = osp.join(all_data_root, 'dataset/instance_' + dataset)
    >>> ensure_dir(data_root)
    >>> io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(data_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(data_root, 'indices.pkl'), indices)
    >>> io.save_pickle(osp.join(data_root, 'pos_edges.pkl'), pos_edges)
