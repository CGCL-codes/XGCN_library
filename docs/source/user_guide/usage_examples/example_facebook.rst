Example: facebook
======================

Let's begin with a social network dataset: facebook. 
It is very small (only contains 4039 nodes and 88234 edges) and is suitable for API tasting! 
If a powerful server is not available, you can just process it and run models on your laptop. 
The data is included in the XGCN repository: ``data/raw_facebook/facebook_combined.txt``. 
You can also download it from SNAP: http://snap.stanford.edu/data/facebook_combined.txt.gz .

---------------------
Data Preparation
---------------------

Before getting started
-------------------------

We recommend to arrange the data with a clear directory structure. 
Before getting started, you may manually setup an ``XGCN_data`` (or other names you like) directory as follows: 
(It's recommended to put your ``XGCN_data`` somewhere else than in this repository.)

.. code:: 

    XGCN_data
    └── dataset
        └── raw_facebook
            └── facebook_combined.txt

We'll use this directory to hold all the different datasets 
and models outputs. 
We refer to its path as ``all_data_root`` in our scripts. 


Evaluation sets generation
----------------------------

At first, we only have the graph data: ``facebook_combined.txt``, and want to generate a validation set and a test set. 
This can be done by using the ``XGCN.data.process.evaluation_set_generation`` module. 

Suppose we are going to generate a 'one_pos_k_neg' evaluation set for fast validation. 
We sample 500 positive edges and 300 negative nodes for each source node. 
The script is as follows: 

.. code:: shell

    # set to your own path:
    file_input_graph='/home/xxx/XGCN_data/dataset/raw_facebook/facebook_combined.txt'
    graph_type='homo'
    graph_format='edge_list'

    seed=1999               # random seed
    num_edge_samples=500    # number of edges to split
    min_src_out_degree=3    # guarantee the minimum out-degree of a source node after the split
    min_dst_in_degree=3     # guarantee the minimum in-degree of a destination node after the split

    # available evaluation_method: 'one_pos_k_neg', 'one_pos_whole_graph', 'multi_pos_whole_graph'
    eval_method='one_pos_k_neg'
    num_neg=300  # the num_neg argument is required when the eval_method='one_pos_k_neg'

    # the output graph will be saved as a text file in edge list format
    # set to your own path:
    file_output_graph='/home/xxx/XGCN_data/dataset/raw_facebook/train.txt'
    file_output_eval_set='/home/xxx/XGCN_data/dataset/raw_facebook/val-one_pos_k_neg.txt'

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

After running the script above, you'll get two new files: ``train.txt`` and ``val-one_pos_k_neg.txt``: 

.. code:: 

    XGCN_data
    └── dataset
        └── raw_facebook
            ├── facebook_combined.txt
            ├── train.txt
            └── val-one_pos_k_neg.txt

We further split some edges in the ``train.txt`` and generate a test set: 

.. code:: shell

    # set to your own path:
    file_input_graph='/home/xxx/XGCN_data/dataset/raw_facebook/facebook_combined.txt'
    graph_type='homo'
    graph_format='edge_list'

    seed=2000               # random seed
    num_edge_samples=8000   # number of edges to split
    min_src_out_degree=3    # guarantee the minimum out-degree of a source node after the split
    min_dst_in_degree=3     # guarantee the minimum in-degree of a destination node after the split

    # available evaluation_method: 'one_pos_k_neg', 'one_pos_whole_graph', 'multi_pos_whole_graph'
    eval_method='multi_pos_whole_graph'
    # num_neg=300  # the num_neg argument is required when the eval_method='one_pos_k_neg'

    # the output graph will be saved as a text file in edge list format
    # set to your own path:
    file_output_graph='/home/xxx/XGCN_data/dataset/raw_facebook/train.txt'
    file_output_eval_set='/home/xxx/XGCN_data/dataset/raw_facebook/test-multi_pos_whole_graph.txt'

    python -m XGCN.data.process.evaluation_set_generation \
        --file_input_graph $file_input_graph \
        --file_output_graph $file_output_graph \
        --file_output_eval_set $file_output_eval_set \
        --seed $seed --graph_type $graph_type --graph_format $graph_format \
        --num_edge_samples $num_edge_samples \
        --min_src_out_degree $min_src_out_degree \
        --min_dst_in_degree $min_dst_in_degree \
        --eval_method $eval_method \
    #    --num_neg $num_neg \

This time we use the 'multi_pos_whole_graph' evaluation method and split 8000 edges 
for fine-grained testing. 
The output 'train.txt' will overwrite the one before, so finally we get three files: 
``train.txt``, ``val-one_pos_k_neg.txt``, and ``test-multi_pos_whole_graph.txt``: 

.. code:: 

    XGCN_data
    └── dataset
        └── raw_facebook
            ├── facebook_combined.txt
            ├── train.txt
            ├── val-one_pos_k_neg.txt
            └── test-multi_pos_whole_graph.txt


Dataset instance generation
-----------------------------

Now we have the complete tran/val/test text data, and is ready to process them into a dataset instance. 

First, let's process the graph (**remember to change the paths in the scripts into your own**): 

.. code:: shell

    ###### process graph for training
    # set to your own path:
    file_input_graph='/home/xxx/XGCN_data/dataset/raw_facebook/train.txt'
    data_root='/home/xxx/XGCN_data/dataset/instance_facebook'
    
    mkdir -p $data_root  # make sure to setup the directory

    graph_type='homo'
    graph_format='edge_list'

    python -m XGCN.data.process.process_int_graph \
        --file_input_graph $file_input_graph --data_root $data_root \
        --graph_type $graph_type --graph_format $graph_format \

Next, we process the validation set and the test set:

.. code:: shell

    ###### process validation set
    file_input='/home/xxx/XGCN_data/dataset/raw_facebook/val-one_pos_k_neg.txt'
    file_output='/home/xxx/XGCN_data/dataset/instance_facebook/val-one_pos_k_neg.pkl'

    evaluation_method='one_pos_k_neg'

    python -m XGCN.data.process.process_evaluation_set \
        --file_input $file_input --file_output $file_output \
        --evaluation_method $evaluation_method \

    ###### process test set
    file_input='/home/xxx/XGCN_data/dataset/raw_facebook/test-multi_pos_whole_graph.txt'
    file_output='/home/xxx/XGCN_data/dataset/instance_facebook/test-multi_pos_whole_graph.pkl'

    evaluation_method='multi_pos_whole_graph'

    python -m XGCN.data.process.process_evaluation_set \
        --file_input $file_input --file_output $file_output \
        --evaluation_method $evaluation_method \

If you have done the above steps successfully, your data directory will look like this: 

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_facebook
        |   ├── facebook_combined.txt
        |   ├── train.txt
        |   ├── val-one_pos_k_neg.txt
        |   └── test-multi_pos_whole_graph.txt
        └── instance_facebook
            ├── info.yaml
            ├── indices.pkl
            ├── indptr.pkl
            ├── val-one_pos_k_neg.pkl
            └── test-multi_pos_whole_graph.pkl

Congratulations! Now we have a complete dataset instance, and is ready to run any models in XGCN!

---------------------
Model Running
---------------------
