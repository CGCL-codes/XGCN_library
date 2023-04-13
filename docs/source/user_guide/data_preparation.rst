Data Preparation
======================

In this section, we introduce how to process your text data into XGCN's "dataset instances". 
For example, suppose you have a dataset looks like this: 

.. code:: 

    raw_facebook
    ├── graph.txt   # edge list
    ├── val.txt     # samples for validation
    └── test.txt    # samples for test

By using XGCN's data processing API, the data are loaded and saved into: 

.. code:: 

    instance_facebook
    ├── info.yaml      # contains some basic information such as "graph type" and "number of nodes"
    ├── indptr.pkl     # graph in CSR format (numpy array)
    ├── indices.pkl
    ├── val.pkl        # evaluation sets
    └── test.pkl

We refer to such processed dataset as "dataset instance", which is required before running XGCN models. 

In the following sections we'll introduce:

* **Graph Processing:** input format of the graph, and the processing API.
* **Evaluation Set Processing:** input format of the evaluation set, and the processing API. If you only have the graph data and do not have evaluation sets, XGCN can also help you generate them. 


---------------------
Graph Processing
---------------------

Input format
--------------------

XGCN specifies integers as standard input node IDs. 
The ID must start from zero. For user-item graphs, both the user IDs and the item IDs are supposed to 
start from zero. 

XGCN specifies edge list as standard input graph structure. The edges are treated as directed. 
In the text file, each line has two nodes: (source node, destination node), and they are seperated by a blank, for example: 

.. code:: 

    0 1
    0 2
    1 4
    2 1
    2 3
    2 4

We also support adjacency list as input, for example: 

.. code:: 

    0 1 2
    1 4
    2 1 3 4
    ...

In this case, each line represents: (source node, neighbor node 1, neighbor node 2, ...). 
Two nodes are also seperated by a blank. 


Processing module
--------------------

The corresponding data processing module is ``XGCN.data.process.process_int_graph``. 
An example of the shell script is as follows: 

.. code:: shell

    # input text file:
    file_input_graph='/home/xxx/xxx/graph.txt'
    # data instance root:
    data_root='/home/xxx/XGCN_data/dataset/instance_xxx'
    
    mkdir -p $data_root  # make sure to setup the directory

    # available graph type: 'homo', 'user-item'
    graph_type='homo'
    # available graph format: 'edge_list', 'adjacency_list'
    graph_format='edge_list'

    python -m XGCN.data.process.process_int_graph \
        --file_input_graph $file_input_graph --data_root $data_root \
        --graph_type $graph_type --graph_format $graph_format \

There are 4 arguments: 

* ``file_input_graph``: the input text file. 
* ``data_root``: the output root (i.e. data instance root). 
* ``graph_type``: available graph type: 'homo' (for homogeneous) or 'user-item'. 
* ``graph_format``: available graph format: 'edge_list' or 'adjacency_list'. 

After running this module, your data root will be like: 

.. code:: 

    instance_xxx
    ├── info.yaml      # contains some basic information such as "graph type" and "number of nodes"
    ├── indptr.pkl     # graph in CSR format (numpy array)
    └── indices.pkl


-----------------------------
Evaluation Set Processing
-----------------------------

Input format
--------------------

We support three kinds of model evaluation method:

* "one_pos_k_neg"

* "one_pos_whole_graph"

* "multi_pos_whole_graph"

They are explained as follows: 

In link prediction tasks, A single evaluation sample can be formulated as: 
(src, pos[1], ..., pos[m], neg[1], ... neg[k]), where src, pos, neg denotes source node, 
positive node, and negative node, respectively. 
The positive nodes usually comes from the removed edges from the original graph. 
The negative nodes are usually sampled from un-interacted nodes 
(i.e. nodes that are not neighbors of the source node). 

For "one_pos_k_neg", each evaluation sample has one positive node and k negative nodes. 
Different evaluation samples may have the same source node. 
The input text file should have N lines and (2+k) columns, two nodes are seperated by a blank: 

.. code:: 

    0 1 33 102 56
    0 2 150 98 72
    2 4 203 42 11
    2 3 34 63 19
    2 5 23 67 48
    5 0 64 130 10

The first column contains the source nodes, the second column cotains the positive nodes, 
and the rest columns are the negative nodes. 

For "one_pos_whole_graph", each evaluation sample has one positive node. 
All the un-interacted nodes in the graph are considered as negative samples. 
Different evaluation samples may have the same source node. 
The input text file should be a N*2 array, and two nodes are seperated by a blank, for example: 

.. code:: 

    0 1
    0 2
    2 4
    2 3
    2 5
    5 0

Each line is a postive pair. 
The first column contains the source nodes, and the second column cotains the positive nodes. 

For "multi_pos_whole_graph", we also consider all the un-interacted nodes as negative samples. 
Each evaluation sample has one or more positive nodes. 
Different evaluation samples should have different source nodes.
The input text file should be an adjacency list, two nodes are seperated by a blank: 

.. code:: 

    0 1 2
    2 4 3 5
    5 0

The first line contains source nodes. Each source should have at least one positive node. 


Processing module
--------------------

The corresponding data processing module is ``XGCN.data.process.process_evaluation_set``. 
An example of the shell script is as follows: 

.. code:: shell

    file_input='/home/xxx/xxx/test.txt'
    file_output='/home/xxx/XGCN_data/dataset/instance_xxx/test.pkl'

    # available evaluation_method: 'one_pos_k_neg', 'one_pos_whole_graph', 'multi_pos_whole_graph'
    evaluation_method='multi_pos_whole_graph'

    python -m XGCN.data.process.process_evaluation_set \
        --file_input $file_input --file_output $file_output \
        --evaluation_method $evaluation_method \

There are 3 arguments: 

* ``file_input``: the input text file. 
* ``file_output``: the output file. We save the data object using ``Pickle``, so it's recommended to name the output as 'xxx.pkl'. 
* ``evaluation_method``: available evaluation method: 'one_pos_k_neg', 'one_pos_whole_graph', and 'multi_pos_whole_graph'. 


Evaluation Set Generation
-----------------------------

Suppose we only have the graph data: 'graph.txt', and want to generate some evaluation sets, 
then we can use the ``XGCN.data.process.evaluation_set_generation`` module: 

.. code:: bash

    file_input_graph='/home/xxx/graph.txt'
    # available graph type: 'homo', 'user-item'
    graph_type='homo'
    # available graph format: 'edge_list', 'adjacency_list'
    graph_format='edge_list'

    seed=1999               # random seed
    num_edge_samples=10000  # number of edges to split
    min_src_out_degree=3    # guarantee the minimum out-degree of a source node after the split
    min_dst_in_degree=3     # guarantee the minimum in-degree of a destination node after the split

    # available evaluation_method: 'one_pos_k_neg', 'one_pos_whole_graph', 'multi_pos_whole_graph'
    eval_method='one_pos_k_neg'
    num_neg=999  # the num_neg argument is required when the eval_method='one_pos_k_neg'

    # the output graph will be saved as a text file in edge list format
    file_output_graph='/home/xxx/graph-1.txt'
    file_output_eval_set='/home/xxx/val.txt'

    python -m XGCN.data.process.evaluation_set_generation \
        --file_input_graph $file_input_graph \
        --file_output_graph $file_output_graph \
        --file_output_eval_set $file_output_eval_set \
        --seed $seed --graph_type $graph_type --graph_format $graph_format \
        --num_edge_samples $num_edge_samples \
        --min_src_out_degree $min_src_out_degree \
        --min_dst_in_degree $min_dst_in_degree \
        --eval_method $eval_method \
        --num_neg $num_neg \

The arguments are:

* ``file_input_graph``: the input text file. 
* ``graph_type``: available graph type: 'homo' (for homogeneous) or 'user-item'. 
* ``graph_format``: available graph format: 'edge_list' or 'adjacency_list'. 
* ``seed``: random seed for edges split. 
* ``num_edge_samples``: number of edges to split. 
* ``min_src_out_degree``: to guarantee the minimum out-degree of a source node after the split. 
* ``min_dst_in_degree``: to guarantee the minimum in-degree of a destination node after the split. 
* ``eval_method``: evaluation method: 'one_pos_k_neg', 'one_pos_whole_graph', and 'multi_pos_whole_graph'. 
* ``num_neg``: number of negative samples for a source node, this argument is required when eval_method='one_pos_k_neg'. 
* ``file_output_graph``: the output graph, which will be saved as a text file in the edge list format. 
* ``file_output_eval_set``: the output text file of the evaluation set. 

You can successively use this module to generate several different evaluation sets. 
The output evaluation sets can then be fed into the ``XGCN.data.process.process_evaluation_set`` module. 
And the final version of the graph for training can be fed into the ``XGCN.data.process.process_int_graph`` module 
to generate a complete dataset instance. 
