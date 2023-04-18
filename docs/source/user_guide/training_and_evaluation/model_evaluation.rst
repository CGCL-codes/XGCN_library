.. _user_guide-training_and_evaluation-model_evaluation:

Model Evaluation
======================

Evaluation methods
----------------------

XGCN support three kinds of model evaluation methods:

* "one_pos_k_neg"

* "one_pos_whole_graph"

* "multi_pos_whole_graph"

XGCN receives text files of the evaluation sets as input and processes them into ``.pkl`` files (please refer to :ref:`Data Preparation <user_guide-data_preparation>`). 
The three evaluation methods are explained as follows: 

In link prediction tasks, A single evaluation sample can be formulated as: 
(src, pos[1], ..., pos[m], neg[1], ... neg[k]), where src, pos, neg denotes source node, 
positive node, and negative node, respectively. 
The positive nodes usually come from the removed edges from the original graph. 
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


Evaluation metrics
-------------------

XGCN supports NDCG\@k and Recall\@k metrics.

Available NDCG\@k metrics are: 
NDCG@20 (write as "n20"), NDCG@50 ("n50"), NDCG@100 ("n100"), and NDCG@300 ("n300"). 

Available Recall\@k metrics are: 
Recall@20 (write as "r20"), Recall@50 ("r50"), Recall@100 ("r100"), and Recall@300 ("r300"). 

To use customized metrics, please modify the ``one_pos_metrics()`` and ``multi_pos_metrics()`` function in ``XGCN/utils/metric.py``. 
More friendly metrics APIs are coming in the future versions. 


Evaluation API
-------------------

To test a model, you can just call ``model.test()``, 
it executes the default testing setting configurations in ``config``: 

.. code:: python

    config = {
        'data_root': ..., 'results_root': ..., 
        'model': 'xGCN', 'seed': 1999, 
        ...,
        'test_method': 'OnePosKNeg_Evaluator', 
        'test_batch_size': 256, 'file_test_set': ...,
        ...
    }
    model = XGCN.create_model(config)
    model.fit()  
    results = model.test()

Or you can specify other test sets:

.. code:: python

    test_config = {
        'test_method': 'multi_pos_whole_graph',
        'test_batch_size': 256,
        'file_test_set': ... 
    }
    results = model.test(test_config)

The function receives a Dict containing three arguments: 

* ``test_method``: specifices the evaluation method. Available options: 'one_pos_k_neg', 'one_pos_whole_graph', and 'multi_pos_whole_graph'. 

* ``test_batch_size``: specifices the batch size. 

* ``file_test_set``: specifices the file of the processed evaluation set. 
