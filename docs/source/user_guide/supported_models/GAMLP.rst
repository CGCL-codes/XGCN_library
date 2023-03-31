GAMLP
==========

Introduction
-----------------

`\[paper\] <https://dl.acm.org/doi/10.1145/3534678.3539121>`_

**Title:** Graph Attention Multi-Layer Perceptron

**Authors:** Wentao Zhang, Ziqi Yin, Zeang Sheng, Yang Li, Wen Ouyang, Xiaosen Li, Yangyu Tao, Zhi Yang, Bin Cui

**Abstract:** Graph neural networks (GNNs) have achieved great success in many graph-based applications. However, the enormous size and high sparsity level of graphs hinder their applications under industrial scenarios. Although some scalable GNNs are proposed for large-scale graphs, they adopt a fixed K-hop neighborhood for each node, thus facing the over-smoothing issue when adopting large propagation depths for nodes within sparse regions. To tackle the above issue, we propose a new GNN architecture --- Graph Attention Multi-Layer Perceptron (GAMLP), which can capture the underlying correlations between different scales of graph knowledge. We have deployed GAMLP in Tencent with the Angel platform, and we further evaluate GAMLP on both real-world datasets and large-scale industrial datasets. Extensive experiments on these 14 graph datasets demonstrate that GAMLP achieves state-of-the-art performance while enjoying high scalability and efficiency. Specifically, it outperforms GAT by 1.3% regarding predictive accuracy on our large-scale Tencent Video dataset while achieving up to 50x training speedup. Besides, it ranks top-1 on both the leaderboards of the largest homogeneous and heterogeneous graph (i.e., ogbn-papers100M and ogbn-mag) of Open Graph Benchmark.

Running with XGCN
----------------------

XGCN implements two version of GAMLP: (1) ``GAMLP``: freeze node embeddings as fixed features;
(2) ``GAMLP_learnable_emb``: has learnable node embeddings. 

**Configuration template for GAMLP:**

.. code:: yaml

    ####### GAMLP-config.yaml #######

    # Dataset/Results root
    data_root: ""
    results_root: ""

    # Trainer configuration
    epochs: 200
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20

    # DataLoader configuration
    Dataset_type: NodeListDataset
    num_workers: 1
    NodeListDataset_type: LinkDataset
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    epoch_sample_ratio: 0.1

    # Evaluator configuration
    val_evaluator: ""
    val_batch_size: 256
    file_val_set: ""
    test_evaluator: ""
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: GAMLP
    seed: 1999
    device: 'cuda:0'

    from_pretrained: 1
    file_pretrained_emb: ''
    freeze_emb: 1

    L2_reg_weight: 0.0
    dnn_lr: 0.001

    GAMLP_type: GAMLP_JK
    num_gcn_layers: 2
    hidden: 512
    n_layers_1: 4
    n_layers_2: 4
    pre_process: 0
    residual: 0
    bns: 0

    loss_fn: bpr


**Run GAMLP from CMD:**

.. code:: bash

    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=GAMLP
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_evaluator WholeGraph_MultiPos_Evaluator --val_batch_size 256 \
        --file_val_set $data_root/val_set.pkl \
        --test_evaluator WholeGraph_MultiPos_Evaluator --test_batch_size 256 \
        --file_test_set $data_root/test_set.pkl \
        --file_pretrained_emb $file_pretrained_emb \

**Configuration template for GAMLP_learnable_emb:**

.. code:: yaml

    ####### GAMLP_learnable_emb-config.yaml ####### 
    
    # Dataset/Results root
    data_root: ""
    results_root: ""

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
    train_batch_size: 1024
    epoch_sample_ratio: 0.1

    # Evaluator configuration
    val_evaluator: ""
    val_batch_size: 256
    file_val_set: ""
    test_evaluator: ""
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: GAMLP_learnable_emb
    seed: 1999

    graph_device: "cuda:0"
    emb_table_device: "cuda:0"
    gnn_device: "cuda:0"
    out_emb_table_device: "cuda:0"

    forward_mode: sample

    emb_dim: 64
    emb_lr: 0.005
    GAMLP_type: GAMLP_JK
    gnn_lr: 0.001
    emb_init_std: 0.1
    use_sparse: 0
    freeze_emb: 0
    from_pretrained: 1
    file_pretrained_emb: ''

    GAMLP_type: GAMLP_JK
    hidden: 512
    n_layers_1: 4
    n_layers_2: 4
    pre_process: 0
    residual: 0
    bns: 0
    dnn_lr: 0.001

    L2_reg_weight: 0.0
    loss_type: bpr

**Run GAMLP from CMD:**

.. code:: bash

    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=GAMLP_learnable_emb
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_evaluator WholeGraph_MultiPos_Evaluator --val_batch_size 256 \
        --file_val_set $data_root/val_set.pkl \
        --test_evaluator WholeGraph_MultiPos_Evaluator --test_batch_size 256 \
        --file_test_set $data_root/test_set.pkl \
        --from_pretrained 1 \
        --file_pretrained_emb $file_pretrained_emb \
