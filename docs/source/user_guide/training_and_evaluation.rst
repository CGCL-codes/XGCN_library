Training and Evaluation
============================

Once the dataset instance is generated (for dataset preprocessing, please refer to previous sections), 
you can run models in ``XGCN/model`` from two ways: command line (CMD) and API functions. 
In this section, we first introduce how to set model configurations, and then present the two ways to run a model. 

Model Configuration
----------------------------

The model configuration in XGCN is basically a Dict containing all the setting parameters. 
XGCN supports parsing model configurations from command line arguments and ``.yaml`` files. 
You can also manually write a Dict with all the parameters in a python script. 

Directory ``config/`` includes ``.yaml`` configuration file templates for all the models. 
A typical ``.yaml`` configuration file including all the arguments is like follows:

.. code:: yaml

    # Dataset/Results root
    data_root: /home/xxx/XGCN_data/dataset/instance_facebook
    results_root: /home/xxx/XGCN_data/model_output/facebook/GraphSAGE/[0]

    # Trainer configuration
    epochs: 200
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20
    
    # DataLoader configuration
    Dataset_type: BlockDataset
    num_workers: 0
    num_gcn_layers: 2
    train_num_layer_sample: "[10, 10]"
    NodeListDataset_type: LinkDataset
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 2048
    epoch_sample_ratio: 0.1

    # Evaluator configuration
    val_evaluator: WholeGraph_MultiPos_Evaluator
    val_batch_size: 256
    file_val_set: /home/xxx/XGCN_data/dataset/instance_facebook/val_set.pkl
    test_evaluator: WholeGraph_MultiPos_Evaluator
    test_batch_size: 256
    file_test_set: /home/xxx/XGCN_data/dataset/instance_facebook/test_set.pkl
    
    # Model configuration
    model: GraphSAGE
    seed: 1999
    forward_mode: sample
    graph_device: cuda
    emb_table_device: cuda
    gnn_device: cuda
    out_emb_table_device: cuda
    from_pretrained: 0
    file_pretrained_emb: ""
    freeze_emb: 0
    use_sparse: 0
    emb_dim: 64 
    emb_init_std: 0.1
    emb_lr: 0.01
    gnn_arch: "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]"
    gnn_lr: 0.01
    loss_type: bpr
    L2_reg_weight: 0.0
    infer_num_layer_sample: "[]"

It consists of five parts:

(1) **Dataset/Results root**. 
Specifies the dataset root and the directory to save the outputs during model training. 

(2) **Trainer configuration**. 
Specifies the configuration about training loop control, e.g. ``epochs``. 

(3) **DataLoader configuration**. 
Specifies the dataloader for training. 

(4) **Evaluator configuration**. 
Specifies the evaluation method and evaluation sets. 

(5) **Model configuration**. 
Specifies the model configuration. 


Run from CMD
------------------

Running models from CMD with the ``XGCN.main.run_model`` module is pretty easy: 

.. code:: bash

    # run model from CMD
    python -m XGCN.main.run_model \
        --config_file "../config/GraphSAGE/config.yaml" \
        --seed 1999 \
        ...

If you want to use a ``.yaml`` configuration file, specify the path 
with the command line argument ``--config_file``. 
Note that a ``.yaml`` file is not a necessity of running the code and has lower 
priority when the same command line argument is given. 


Run from API
------------------

XGCN provide API functions to build and train a model, for example: 

.. code:: python

    config = {'model': 'xGCN', 'seed': 1999, ... }
    # configurations parsed from command line arguments or .yaml file
    
    data = {}
    # a dict is needed for holding some global data objects:
    
    # build the modules:
    model = XGCN.build_Model(config, data)

    train_dl = XGCN.build_DataLoader(config, data)

    val_evaluator = XGCN.build_val_Evaluator(config, data, model)
    test_evaluator = XGCN.build_test_Evaluator(config, data, model)

    trainer = XGCN.build_Trainer(config, data, model, train_dl,
                                 val_evaluator, test_evaluator)
    
    # start training and test the model after the training process has converged
    trainer.train_and_test()

The ``trainer`` will train the model utill convergence and automatically save the best model 
on the validation set. 
For more information, you can referance ``XGCN/main/run_model.py``.


Model Running Examples
--------------------------

We provide model running examples on two large-scale social network dataset: Pokec and LiveJournal, 
which are used in our xGCN paper.

The data can be downloaded from here: 
`pokec_and_livejournal_data <https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip>`_. 
(The Xbox dataset is an industrial one and is not able to be public.) 
To download and process them, please refer to the "Data Preparation" section and 
the scripts in ``script/data_process/pokec_and_livejournal``. 

Please refer to ``script/model`` which includes all the scripts for different 
datasets (The used datasets in our paper are: Pokec, LiveJournal, and Xbox. 
Note that we do not tune models on the facebook dataset example): 

.. code:: 

    script
    └── model
        ├─ GAMLP
        ├─ ...
        └─ xGCN
           ├─ run_xGCN-facebook.sh
           ├─ run_xGCN-livejournal.sh
           ├─ run_xGCN-pokec.sh
           └─ run_xGCN-xbox-3m.sh

To run a model, you only need to modify the ``all_data_root`` and ``config_file_root`` 
arguments in the script to your own paths. 


Model Evaluation
--------------------

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
