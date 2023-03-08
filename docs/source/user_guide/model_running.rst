Model Training
=========================

Once the data preparation is done, users can easily run a model 
with the ``XGCN.main.run_model`` module: 

.. code:: bash

    python -m XGCN.main.run_model \
        --model "GraphSAGE" \
        --seed 1999 \
        --data_root ... \
        --results_root ... \
        ...

In this section, we introduce the model configuration and training results 
using the facebook dataset created in the previous subsection 
and the common model GraphSAGE as an example.


Configuration parsing
----------------------------

The ``XGCN.main.run_model`` module supports parsing model configurations 
from command line arguments and ``.yaml`` files. 
Directory ``config/`` includes ``.yaml`` configuration file templates for all the models, 
and directory ``scripts/`` provides ``.sh`` shell scripts to run the models. 

If you want to use a ``.yaml`` configuration file, specify the path 
with the command line argument ``--config_file`` like follows:

.. code:: bash

    python -m XGCN.main.run_model \
        --config_file "../config/GraphSAGE/config.yaml" \
        --seed 1999 \
        ...

Note that a ``.yaml`` file is not a necessity of running the code and has lower 
priority when the same command line argument is given. 


Configuration components 
-------------------------------

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

.. 3. Training Data and Results
.. -----------------------------


.. In the last section, we process the raw facebook data and generate a dataset instance:

..     XGCN_data
..     └── dataset
..         └── instance_facebook
..             ├── indices.pkl
..             ├── indptr.pkl
..             ├── info.yaml
..             ├── pos_edges.pkl
..             ├── test_set.pkl
..             └── val_set.pkl

.. With these cached data, we can run all the models by specifying the ``data_root`` in the configuration, 
.. which is ``/xxx/XGCN_data/dataset/instance_facebook`` here. 
.. We use the


Training Process
-----------------------


