Training and Evaluation
============================

Once the dataset instance is generated (for dataset preprocessing, please refer to previous sections), 
you can run models in two ways: Command Line Interface (CLI) or API functions. 
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
    data_root: ""
    results_root: ""

    # Trainer configuration
    epochs: 200
    use_validation_for_early_stop: 1
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20
    val_evaluator: ""
    val_batch_size: 256
    file_val_set: ""

    # Testing configuration  (required for model.test())
    test_evaluator: ""
    test_batch_size: 256
    file_test_set: ""

    # DataLoader configuration
    Dataset_type: BlockDataset
    num_workers: 0
    num_gcn_layers: 2
    train_num_layer_sample: "[10, 20]"
    NodeListDataset_type: LinkDataset
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    epoch_sample_ratio: 0.1

    # Model configuration
    model: GraphSAGE
    seed: 1999

    graph_device: "cuda:0"
    emb_table_device: "cuda:0"
    gnn_device: "cuda:0"
    out_emb_table_device: "cuda:0"

    forward_mode: sample
    infer_num_layer_sample: "[10, 20]"

    from_pretrained: 0
    file_pretrained_emb: ""
    freeze_emb: 0
    use_sparse: 0
    emb_dim: 64 
    emb_init_std: 0.1
    emb_lr: 0.005

    gnn_arch: "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]"
    gnn_lr: 0.01

    loss_type: bpr
    L2_reg_weight: 0.0


It consists of five parts:

(1) **Dataset/Results root**. 
Specifies the dataset root and the directory to save the outputs during the model training. 

(2) **Trainer configuration**. 
Specifies the configuration about training loop control, e.g. ``epochs``. 

(3) **Testing configuration**. 
Specifies the configurations about model testing. This field is required for ``model.test()`` function. 

(4) **DataLoader configuration**. 
Specifies the dataloader for training. 

(5) **Model configuration**. 
Specifies the model configuration such as hyper-parameters. 


Run from CLI
------------------

Running models from CLI with the ``XGCN.main.run_model`` module is pretty easy: 

.. code:: bash

    # run model from CLI
    python -m XGCN.main.run_model \
        --config_file "../config/GraphSAGE/config.yaml" \
        --seed 1999 \
        --data_root ... \
        --results_root ... \

If you want to use a ``.yaml`` configuration file, specify the path 
with the command line argument ``--config_file``. 
Note that a ``.yaml`` file is not a necessity of running the code and has lower 
priority when the same command line argument is given. 

We provide shell scripts examples for all the models, **please refer to...**.

Once a model is trained, the output data will be saved at ``results_root``: 

.. code:: 

    XGCN_data
    └── model_output
        └── facebook
            └── xGCN
                └── [seed0]
                    ├── config.yaml             # configurations of the running
                    ├── mean_time.json          # time consumption information in seconds
                    ├── test_results.json       # test results
                    ├── train_record_best.json  # validation results of the best epoch
                    ├── train_record.txt        # validation results and losses during training
                    └── model                   # a directory containing the saved model


Run from API functions
--------------------------

XGCN provides API functions to create and train a model, for example: 

.. code:: python

    import XGCN

    # configurations parsed from command line arguments or a .yaml file
    config = {
        'data_root': ..., 'results_root': ..., 
        'model': 'xGCN', 'seed': 1999, 
        ...,
        'test_evaluator': 'OnePosKNeg_Evaluator', 
        'test_batch_size': 256, 'file_test_set': ...,
        ...
    }
    model = XGCN.create_model(config)

    model.fit()              # model training, 
                             # the best model on the validation set 
                             # will be saved at results_root

After training, models can be evaluated on one or more test sets 
(for more information about model evaluation, **please refer to...**): 

.. code:: python

    # model testing (default settings in config)
    results = model.test()

    # testing on other test sets
    test_config_2 = {
        'test_evaluator': 'WholeGraph_MultiPos_Evaluator',
        'test_batch_size': 256,
        'file_test_set': ...  # another test set
    }
    results2 = model.test(test_config_2)

XGCN provides some model inference APIs 
(for more information, **please refer to...**): 

.. code:: python

    # infer scores given a source node and one or more target nodes:
    target_score = model.infer_target_score(
        src=5, 
        target=torch.LongTensor(101, 102, 103)
    )

    # infer top-k recommendations for a source node
    score, topk_node = model.infer_topk(k=100, src=5, mask_nei=True)

    # save the output embeddings as a text file
    model.save_emb_as_txt(filename='out_emb_table.txt')


XGCN also supports load pretrained models to do fine-tunning: 

.. code:: python

    config = io.load_yaml(...)  # the previously saved config.yaml
    config['emb_lr'] = 0.0001   # change some hyper-paramenters

    model = XGCN.create_model(config)
    model.load()  # load the saved model      
    model.fit()   # training on the hyper-paramenters
    new_resutls = model.test()


Model Evaluation
--------------------

To test a model, you can just call ``model.test()``, 
it executes the default testing setting configurations in ``config``: 

.. code:: python

    config = {
        'data_root': ..., 'results_root': ..., 
        'model': 'xGCN', 'seed': 1999, 
        ...,
        'test_evaluator': 'OnePosKNeg_Evaluator', 
        'test_batch_size': 256, 'file_test_set': ...,
        ...
    }
    model = XGCN.create_model(config)
    model.fit()  
    results = model.test()

Or you can specify other test sets

.. code:: python

    test_config = {
        'test_evaluator': 'WholeGraph_MultiPos_Evaluator',
        'test_batch_size': 256,
        'file_test_set': ... 
    }
    results = model.test(test_config)

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
