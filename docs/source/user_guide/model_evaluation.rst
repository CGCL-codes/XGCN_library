Model Evaluation
======================


Evaluation sets
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


Evaluation APIs
---------------------
