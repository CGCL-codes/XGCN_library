UltraGCN
==============

`\[paper\] <>`_

**Title:** 

**Authors:** 

**Abstract:** 

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
    model: UltraGCN
    seed: 1999

    file_ultra_constrain_mat: ""
    file_ii_topk_neighbors: ""
    file_ii_topk_similarity_scores: ""

    device: "cuda:0"
    emb_table_device: "cuda:0"

    emb_lr: 0.005
    emb_dim: 64
    emb_init_std: 0.0001
    use_sparse: 1
    freeze_emb: 0
    from_pretrained: 0
    file_pretrained_emb: ''

    num_neg: 128
    neg_weight: 128

    topk: 8
    lambda: 0.8
    gamma: 1.5
    L2_reg_weight: 0.0001
