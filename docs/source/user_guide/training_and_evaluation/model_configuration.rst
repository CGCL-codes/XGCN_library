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

(1) `Dataset/Results root`_

(2) `Trainer configuration`_

(3) `Testing configuration`_

(4) `DataLoader configuration`_

(5) `Model configuration`_


.. _user_guide-training_and_evaluation-data_root_results_root:

Dataset/Results root
---------------------------

This part only has two fields: 

* ``data_root``: The dataset instance root. \
For dataset instance generation, please refer to 
:ref:`Data Preparation <user_guide-data_preparation>`. 

* ``results_root``: The directory to save the outputs during the model training. 
Note that when calling the  ``XGCN.create_model(config)`` function, 
the ``results_root`` directory will be automatically created if it does not exist. 

.. _user_guide-training_and_evaluation-trainer_config:

Trainer configuration
---------------------------

This part specifies the configuration about training loop control: 

epochs: 200
use_validation_for_early_stop: 1
val_freq: 1
key_score_metric: r100
convergence_threshold: 20
val_method: ""
val_batch_size: 256
file_val_set: ""


.. _user_guide-training_and_evaluation-testing_config:

Testing configuration
---------------------------

This part specifies the configurations about model testing. This field is required 
for ``model.test()`` function. 


.. _user_guide-training_and_evaluation-dataloader_config:

DataLoader configuration
---------------------------

This part specifies the dataloader for training. 


.. _user_guide-training_and_evaluation-model_config:

Model configuration
---------------------------

This part specifies the model configuration such as hyper-parameters. 


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
