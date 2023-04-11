PPRGo
===========

Introduction
-----------------

`\[paper\] <https://dl.acm.org/doi/abs/10.1145/3394486.3403296>`_

**Title:** Scaling Graph Neural Networks with Approximate PageRank

**Authors:** Aleksandar Bojchevski, Johannes Gasteiger, Bryan Perozzi, Amol Kapoor, Martin Blais, Benedek Rózemberczki, Michal Lukasik, Stephan Günnemann

**Abstract:** Graph neural networks (GNNs) have emerged as a powerful approach for solving many network mining tasks. However, learning on large graphs remains a challenge -- many recently proposed scalable GNN approaches rely on an expensive message-passing procedure to propagate information through the graph. We present the PPRGo model which utilizes an efficient approximation of information diffusion in GNNs resulting in significant speed gains while maintaining state-of-the-art prediction performance. In addition to being faster, PPRGo is inherently scalable, and can be trivially parallelized for large datasets like those found in industry settings. 
We demonstrate that PPRGo outperforms baselines in both distributed and single-machine training environments on a number of commonly used academic graphs. To better analyze the scalability of large-scale graph learning methods, we introduce a novel benchmark graph with 12.4 million nodes, 173 million edges, and 2.8 million node features. We show that training PPRGo from scratch and predicting labels for all nodes in this graph takes under 2 minutes on a single machine, far outpacing other baselines on the same graph. We discuss the practical application of PPRGo to solve large-scale node classification problems at Google.

Running with XGCN
----------------------

**Configuration template for PPR:**

.. code:: yaml

    ####### PPR-config.yaml #######

    data_root: ""
    results_root: ""

    seed: 1999
    topk: 100
    num_walks: 1000
    walk_length: 30
    alpha: 0.3

**Run PPR from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=PPR
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    python -m XGCN.main.run_ppr --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \


**Configuration template for PPRGo:**

.. code:: yaml

    ####### PPRGo-config.yaml #######

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
    val_method: ""
    val_batch_size: 256
    file_val_set: ""
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: PPRGo
    seed: 1999

    ppr_data_root: ""

    ppr_data_device: "cuda:0"
    emb_table_device: "cuda:0"
    forward_device: "cuda:0"
    out_emb_table_device: "cuda:0"

    from_pretrained: 0
    file_pretrained_emb: ""
    freeze_emb: 0
    use_sparse: 1
    emb_dim: 64 
    emb_init_std: 0.1
    emb_lr: 0.005

    topk: 32
    use_uniform_weight: 1

    loss_type: bpr
    L2_reg_weight: 0.0


**Run PPRGo from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=PPRGo
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method MultiPosWholeGraph_Evaluator --val_batch_size 256 \
        --file_val_set $data_root/val_set.pkl \
        --test_method MultiPosWholeGraph_Evaluator --test_batch_size 256 \
        --file_test_set $data_root/test_set.pkl \
        --from_pretrained 0 --file_pretrained_emb $file_pretrained_emb \
        --ppr_data_root $all_data_root/model_output/$dataset/PPR/[seed0] \
