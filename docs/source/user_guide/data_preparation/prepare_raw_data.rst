Prepare the Raw Data
======================

First of all, let's make an empty directory named ``XGCN_data`` (or any one you like) 
to place all the datasets and model outputs. 
It's recommended to put ``XGCN_data`` somewhere else than in this repository. 

We recommend to arrange the data with a clear directory structure. 
From the beginning, you may manually setup the ``XGCN_data`` directory as follows, 
where ``raw_graph.txt`` is renamed from ``facebook_combined.txt`` for consistency. 

.. code:: 

    XGCN_data
    └── dataset
        └── raw_facebook
            └── raw_graph.txt

After some further data processing and model running, your directory may look like: 

.. code:: 

    XGCN_data
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

XGCN supports two kinds of text graph data as input: 

(1) **Edge-list**. One line represents an edge: (source node, destination node). The nodes are seperated by a whitespace, 
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

The raw facebook data follows the edge-list format. 
