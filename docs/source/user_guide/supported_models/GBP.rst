GBP
========

`\[paper\] <https://arxiv.org/abs/2010.15421>`_

**Title:** Scalable Graph Neural Networks via Bidirectional Propagation

**Authors:** Ming Chen, Zhewei Wei, Bolin Ding, Yaliang Li, Ye Yuan, Xiaoyong Du, Ji-Rong Wen

**Abstract:** Graph Neural Networks (GNN) is an emerging field for learning on non-Euclidean data. Recently, there has been increased interest in designing GNN that scales to large graphs. Most existing methods use "graph sampling" or "layer-wise sampling" techniques to reduce training time. However, these methods still suffer from degrading performance and scalability problems when applying to graphs with billions of edges. This paper presents GBP, a scalable GNN that utilizes a localized bidirectional propagation process from both the feature vectors and the training/testing nodes. Theoretical analysis shows that GBP is the first method that achieves sub-linear time complexity for both the precomputation and the training phases. An extensive empirical study demonstrates that GBP achieves state-of-the-art performance with significantly less training/testing time. Most notably, GBP can deliver superior performance on a graph with over 60 million nodes and 1.8 billion edges in less than half an hour on a single machine. The codes of GBP can be found at https://github.com/chennnM/GBP .

.. code:: yaml

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
    train_batch_size: 2048
    epoch_sample_ratio: 0.1

    # Evaluator configuration
    val_evaluator: ""
    val_batch_size: 256
    file_val_set: ""
    test_evaluator: ""
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: GBP
    seed: 1999
    device: 'cuda:0'

    from_pretrained: 1
    file_pretrained_emb: ''
    freeze_emb: 1

    alpha: 0.1
    walk_length: 6
    rmax_ratio: 0.01
    dnn_arch: "[nn.Linear(64, 1024), nn.Tanh(), nn.Linear(1024, 64)]"
    dnn_lr: 0.001
    L2_reg_weight: 0.0

    loss_fn: bpr
