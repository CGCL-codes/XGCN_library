SSGC
===========

Introduction
-----------------

`\[paper\] <https://openreview.net/forum?id=CYO5T-YjWZV>`_

**Title:** Simple Spectral Graph Convolution

**Authors:** Hao Zhu, Piotr Koniusz

**Abstract:** Graph Convolutional Networks (GCNs) are leading methods for learning graph representations. However, without specially designed architectures, the performance of GCNs degrades quickly with increased depth. As the aggregated neighborhood size and neural network depth are two completely orthogonal aspects of graph representation, several methods focus on summarizing the neighborhood by aggregating K-hop neighborhoods of nodes while using shallow neural networks. However, these methods still encounter oversmoothing, and suffer from high computation and storage costs. In this paper, we use a modified Markov Diffusion Kernel to derive a variant of GCN called Simple Spectral Graph Convolution (SSGC). Our spectral analysis shows that our simple spectral graph convolution used in SSGC is a trade-off of low- and high-pass filter bands which capture the global and local contexts of each node. We provide two theoretical claims which demonstrate that we can aggregate over a sequence of increasingly larger neighborhoods compared to competitors while limiting severe oversmoothing.  Our experimental evaluations show that SSGC with a linear learner is competitive in text and node classification tasks. Moreover, SSGC is comparable to other state-of-the-art methods for node clustering and community prediction tasks.

Running with XGCN
----------------------

**Configuration template for SSGC:**

.. code:: yaml

    ####### SSGC-config.yaml #######

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
    model: SSGC
    seed: 1999

    device: 'cuda:0'

    from_pretrained: 1
    file_pretrained_emb: ''
    freeze_emb: 1

    L2_reg_weight: 0.0
    dnn_lr: 0.001

    num_gcn_layers: 2

    loss_fn: bpr


**Run SSGC from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=SSGC
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
        --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \


**Configuration template for SSGC_learnable_emb:**

.. code:: yaml

    ####### SSGC_learnable_emb-config.yaml #######

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
    val_method: ""
    val_batch_size: 256
    file_val_set: ""
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: SSGC_learnable_emb
    seed: 1999

    graph_device: "cuda:0"
    emb_table_device: "cuda:0"
    gnn_device: "cuda:0"
    out_emb_table_device: "cuda:0"

    forward_mode: sample

    emb_dim: 64
    emb_lr: 0.005
    gnn_lr: 0.001
    emb_init_std: 0.1
    use_sparse: 0
    freeze_emb: 0
    from_pretrained: 1
    file_pretrained_emb: ''

    L2_reg_weight: 0.0
    loss_type: bpr


**Run SSGC_learnable_emb from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=SSGC_learnable_emb
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
        --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \
