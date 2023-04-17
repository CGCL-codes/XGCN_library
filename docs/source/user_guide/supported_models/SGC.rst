.. _supported_models-SGC:

SGC
=========

Introduction
-----------------

`\[paper\] <https://arxiv.org/abs/1902.07153>`_

**Title:** Simplifying Graph Convolutional Networks

**Authors:** Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger

**Abstract:** Graph Convolutional Networks (GCNs) and their variants have experienced significant attention and have become the de facto methods for learning graph representations. GCNs derive inspiration primarily from recent deep learning approaches, and as a result, may inherit unnecessary complexity and redundant computation. In this paper, we reduce this excess complexity through successively removing nonlinearities and collapsing weight matrices between consecutive layers. We theoretically analyze the resulting linear model and show that it corresponds to a fixed low-pass filter followed by a linear classifier. Notably, our experimental evaluation demonstrates that these simplifications do not negatively impact accuracy in many downstream applications. Moreover, the resulting model scales to larger datasets, is naturally interpretable, and yields up to two orders of magnitude speedup over FastGCN.

Running with XGCN
----------------------

**Configuration template for SGC:**

.. code:: yaml

    ####### SGC-config.yaml #######

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
    model: SGC
    seed: 1999

    device: 'cuda:0'

    from_pretrained: 1
    file_pretrained_emb: ''
    freeze_emb: 1

    L2_reg_weight: 0.0
    dnn_lr: 0.001

    num_gcn_layers: 2

    loss_fn: bpr

**Run SGC from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=SGC
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


**Configuration template for SGC_learnable_emb:**

.. code:: yaml

    ####### SGC_learnable_emb-config.yaml #######

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
    train_batch_size: 2048
    epoch_sample_ratio: 0.1

    # Evaluator configuration
    val_method: ""
    val_batch_size: 256
    file_val_set: ""
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: SGC_learnable_emb
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


**Run SGC_learnable_emb from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""
    
    dataset=facebook
    model=SGC_learnable_emb
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
