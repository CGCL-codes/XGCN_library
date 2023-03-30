SimpleX
===========

`\[paper\] <https://dl.acm.org/doi/10.1145/3459637.3482297>`_

**Title:** SimpleX: A Simple and Strong Baseline for Collaborative Filtering

**Authors:** Kelong Mao, Jieming Zhu, Jinpeng Wang, Quanyu Dai, Zhenhua Dong, Xi Xiao, Xiuqiang He

**Abstract:** Collaborative filtering (CF) is a widely studied research topic in recommender systems. The learning of a CF model generally depends on three major components, namely interaction encoder, loss function, and negative sampling. While many existing studies focus on the design of more powerful interaction encoders, the impacts of loss functions and negative sampling ratios have not yet been well explored. In this work, we show that the choice of loss function as well as negative sampling ratio is equivalently important. More specifically, we propose the cosine contrastive loss (CCL) and further incorporate it to a simple unified CF model, dubbed SimpleX. Extensive experiments have been conducted on 11 benchmark datasets and compared with 29 existing CF models in total. Surprisingly, the results show that, under our CCL loss and a large negative sampling ratio, SimpleX can surpass most sophisticated state-of-the-art models by a large margin (e.g., max 48.5% improvement in NDCG@20 over LightGCN). We believe that SimpleX could not only serve as a simple strong baseline to foster future research on CF, but also shed light on the potential research direction towards improving loss function and negative sampling.

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
    num_neg: 256
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
    model: SimpleX
    seed: 1999

    device: "cuda:0"

    neg_weight: 256

    train_num_layer_sample: "[]"

    emb_dim: 64
    emb_init_std: 0.1
    emb_lr: 0.001
    use_sparse: 1
    freeze_emb: 0
    from_pretrained: 0

    margin: 0.4
    theta: 0.5
    L2_reg_weight: 0.0001

    use_uniform_weight: 1
