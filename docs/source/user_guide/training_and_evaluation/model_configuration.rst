.. _user_guide-training_and_evaluation-model_configuration:

Model Configuration
======================

The model configuration in XGCN is basically a Python Dict containing all the setting parameters. 
XGCN supports parsing model configurations from command line arguments and ``.yaml`` files. 
You can also manually write a Dict with all the parameters in a python script. 


.. _user_guide-training_and_evaluation-config_template:

Configuration Template
---------------------------

Directory ``config/`` includes ``.yaml`` configuration file templates for all the models. 
Each file contains **all** the arguments needed to run a model. 
A typical ``.yaml`` configuration file is like this:

.. code:: yaml

    # Dataset/Results root
    data_root: ""       # root of the dataset instance
    results_root: ""    # root for model outputs, training record, and evaluation results

    # Trainer configuration
    epochs: 200
    use_validation_for_early_stop: 1
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20
    val_method: ""
    val_batch_size: 256
    file_val_set: ""

    # Testing configuration  (required for model.test())
    test_method: ""
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
    str_num_total_samples: num_edges
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

The configuration consists of five parts:

(1) :ref:`Dataset/Results root <user_guide-training_and_evaluation-data_root_results_root>`

(2) :ref:`Trainer configuration <user_guide-training_and_evaluation-trainer_config>`

(3) :ref:`Testing configuration <user_guide-training_and_evaluation-testing_config>`

(4) :ref:`DataLoader configuration <user_guide-training_and_evaluation-dataloader_config>`

(5) :ref:`Model configuration <user_guide-training_and_evaluation-model_config>`


.. _user_guide-training_and_evaluation-data_root_results_root:

Dataset/Results root
---------------------------

This part only has two arguments: 

* ``data_root``: (str) The dataset instance root (for dataset instance generation, please refer to :ref:`Data Preparation <user_guide-data_preparation>`). This argument specifies which dataset to use. 

* ``results_root``: (str) The directory to save the outputs during the model training. 

Note that when calling the  ``XGCN.create_model(config)`` function, the ``results_root`` directory will be automatically created if it does not exist. 

.. _user_guide-training_and_evaluation-trainer_config:

Trainer configuration
---------------------------

This part specifies the configuration about training loop control: 

* ``epochs``: (int) The maximum epochs to run.  

* ``use_validation_for_early_stop``: (bool: 0 or 1) Whether to use validation scores for early stop. If this argument is ``1``, then the following 6 arguments are required. 

* ``val_freq``: (int) Evaluate the model on the validation set every ``val_freq`` epochs. 

* ``key_score_metric``: (str) The metric used for early stop. Once a better result on the ``key_score_metric`` is achieved on the validation set, the whole model will be saved. For available metrics, please refer to :ref:`Model Evaluation <user_guide-training_and_evaluation-model_evaluation>`. 

* ``convergence_threshold``: (int) If the ``key_score_metric`` has not increased for ``convergence_threshold`` epochs, then we consider the training has already converged and the early stop is triggered (stop training). 

* ``val_method``: (str) Evaluation method for validation. For evaluation methods, please refer to :ref:`Model Evaluation <user_guide-training_and_evaluation-model_evaluation>`. 

* ``val_batch_size``: (int) Batch size for validation. 

* ``file_val_set``: (str) The file of the validation set. 


.. _user_guide-training_and_evaluation-testing_config:

Testing configuration
---------------------------

Note that this part is optional for model training (i.e. ``model.fit()``) 
and is required for ``model.test()`` function. For more information about testing, 
please refer to :ref:`Model Evaluation <user_guide-training_and_evaluation-model_evaluation>`. 

* ``test_method``: (str) Evaluation method for testing. 

* ``test_batch_size``: (int) Batch size for testing. 

* ``file_test_set``: (str) The file of the test set. 


.. _user_guide-training_and_evaluation-dataloader_config:

DataLoader configuration
---------------------------

In general, we consider two types of dataloader for GNN training: 

(1) **node-only dataloader:** In each mini-batch, returns the needed node IDs: (source nodes, positive nodes, negative nodes). 

(2) **block dataloader:** Not only returns node IDs, but also returns the DGL's "blocks" (also known as "message flow graph" (MFG)). 

The **node-only dataloader** is used in the following cases:  

* The GNN's message-passing is performed on the full graph. i.e. embeddings of all the nodes are inferred in a mini-bach. 

* Additional graph information is not need. For example, the PPRGo model use the top-k PPR neighbor for each node, and the neighbors are held by the model itself. As another example, the UltraGCN model does not use message-passing, the node IDs is enough for batch training. 

The **block dataloader** is used for graph sampling when training on large graphs 
(please refer to `DGL docs: Chapter 6: Stochastic Training on Large Graphs <https://docs.dgl.ai/en/latest/guide/minibatch.html>`_ for more information). 
In each mini-batch, it returns node IDs and the needed DGL "blocks".

For some GNNs, XGCN provide both "full graph message-passing" and "block message-passing" training method. 
Their configuration templates are included in the ``config/`` directory. For example:

.. code::

    config
    ├── LightGCN-full_graph-config.yaml
    ├── LightGCN-block-config.yaml
    ├── GraphSAGE-full_graph-config.yaml
    ├── GraphSAGE-block-config.yaml
    ...

The "full graph message-passing" training uses the node-only dataloader, 
and the "block message-passing" training uses the block dataloader. 

Their configuration arguments of the two dataloaders are as follows: 

.. code:: yaml
    
    ####### for node-only dataloader #######
    # DataLoader configuration
    Dataset_type: NodeListDataset  # fixed
    num_workers: 0
    NodeListDataset_type: LinkDataset  # fixed
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    str_num_total_samples: num_edges
    epoch_sample_ratio: 0.1

.. code:: yaml

    #######  for block dataloader ####### 
    # DataLoader configuration
    Dataset_type: BlockDataset  # fixed
    num_workers: 0
    num_gcn_layers: 2
    train_num_layer_sample: "[10, 20]"
    NodeListDataset_type: LinkDataset  # fixed
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    str_num_total_samples: num_edges
    epoch_sample_ratio: 0.1

The meanings of the arguments are as follows:

* ``Dataset_type``: (str) This argument is fixed as "NodeListDataset" for node-only dataloader, and is fixed as "BlockDataset" for block dataloader. 

* ``NodeListDataset_type``: (str) This field is fix as "LinkDataset". 

* ``num_workers``: (int) Number of workers for dataloading. 0 means loading data in the main process. Set to 0 if the graph is on GPU. 

* ``num_gcn_layers``: (int) Number of GNN(GCN) layers. This argument is required for the block dataloader. 

* ``train_num_layer_sample``: (str) Number of nodes to sample in each layer during training. For example, "[10, 20]" means 10 nodes in the first layer and 20 nodes in the second layer. This argument is required for the block dataloader. 

* ``pos_sampler``: (str) Postive sampler. Available options:
    + **"ObservedEdges_Sampler"**: given edge IDs, return the edges. 
    + **"NodeBased_ObservedEdges_Sampler"**: given node IDs, sample a neighbor for each node. 

* ``neg_sampler``: (str) Negative sampler. Available options: 
    + **"RandomNeg_Sampler"**: random sampling from all the nodes (from all the item nodes for user-item graphs). 
    + **"StrictNeg_Sampler"**: sample strictly un-interacted nodes. 

* ``num_neg``: (int) Number of negative samples for each positive sample. 

* ``str_num_total_samples``: (str) the number of all the IDs used to generate samples. Available options:
    + **"num_edges"**: sample from all the edges for training, this is required by "ObservedEdges_Sampler";
    + **"num_nodes"**: first sample a node, then sample a neighbor from it. This is required by "NodeBased_ObservedEdges_Sampler"; 
    + **"num_users"**: This is required by the "NodeBased_ObservedEdges_Sampler" when the graph is a user-item network. 

* ``epoch_sample_ratio``: (float) the ``str_num_total_samples`` might be a large number, e.g. the edges in a graph. We can shrink the number of samples for an epoch to ``epoch_sample_ratio`` \* ``str_num_total_samples`` by setting ``epoch_sample_ratio`` to a value between 0 and 1. We can also expand the number of samples by setting it larger than 1. 

* ``BatchSampleIndicesGenerator_type``: (str) the way to generate samples IDs in a batch. Available options: 
    + **"SampleIndicesWithReplacement"**: sampling without replacement, e.g. sampling from all the edges without replacement; 
    + **"SampleIndicesWithoutReplacement"**: sampling with replacement, e.g. all the edges is guaranteed to be sampled within a number of epochs. 

* ``train_batch_size``: (int) training batch size. 

.. _user_guide-training_and_evaluation-model_config:

Model configuration
---------------------------

This part specifies the model configuration such as hyper-parameters. 
Please refer to :ref:` <user_guide-supported_models>` for the detailed explaination of each model. 


.. _user_guide-training_and_evaluation-load_config_from_yaml:

Load config from yaml file
---------------------------

We can load a ``.yaml`` configuration file with ``XGCN.data.io`` module:

.. code:: python

    import XGCN
    from XGCN.data import io

    config = io.load_yaml('config.yaml')  # load template
    config['data_root'] = ...             # add/modify some configurations


.. _user_guide-training_and_evaluation-parse_config_from_command_line:

Parse config from command line
--------------------------------

We also provide a ``parse_arguments()`` to parse command line arguments: 

.. code:: python

    import XGCN
    from XGCN.utils.parse_arguments import parse_arguments

    config = parse_arguments()


You can specify a ``.yaml`` configuration file with ``--config_file``. 
Note that a configuration file is not a necessity for the ``parse_arguments()`` function 
and has lower priority when the same command line argument is given. 
