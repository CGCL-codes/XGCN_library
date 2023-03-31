GIN
=======

Introduction
-----------------

`\[paper\] <https://arxiv.org/abs/1810.00826>`_

**Title:** How Powerful are Graph Neural Networks?

**Authors:** Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka

**Abstract:** Graph Neural Networks (GNNs) are an effective framework for representation learning of graphs. GNNs follow a neighborhood aggregation scheme, where the representation vector of a node is computed by recursively aggregating and transforming representation vectors of its neighboring nodes. Many GNN variants have been proposed and have achieved state-of-the-art results on both node and graph classification tasks. However, despite GNNs revolutionizing graph representation learning, there is limited understanding of their representational properties and limitations. Here, we present a theoretical framework for analyzing the expressive power of GNNs to capture different graph structures. Our results characterize the discriminative power of popular GNN variants, such as Graph Convolutional Networks and GraphSAGE, and show that they cannot learn to distinguish certain simple graph structures. We then develop a simple architecture that is provably the most expressive among the class of GNNs and is as powerful as the Weisfeiler-Lehman graph isomorphism test. We empirically validate our theoretical findings on a number of graph classification benchmarks, and demonstrate that our model achieves state-of-the-art performance.

Running with XGCN
----------------------

**Configuration template:**

.. code:: yaml

    ####### GIN-config.yaml #######

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
    train_num_layer_sample: "[10, 20]"
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
    model: GIN
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

    gnn_lr: 0.001

    loss_type: bpr
    L2_reg_weight: 0.0


**Run from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=GIN
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
        --freeze_emb 0 \
