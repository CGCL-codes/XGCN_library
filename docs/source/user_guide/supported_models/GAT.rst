GAT
=========

Introduction
-----------------

`\[paper\] <https://arxiv.org/abs/1710.10903>`_

**Title:** Graph Attention Networks

**Authors:** Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio

**Abstract:** We present graph attention networks (GATs), novel neural network architectures that operate on graph-structured data, leveraging masked self-attentional layers to address the shortcomings of prior methods based on graph convolutions or their approximations. By stacking layers in which nodes are able to attend over their neighborhoods' features, we enable (implicitly) specifying different weights to different nodes in a neighborhood, without requiring any kind of costly matrix operation (such as inversion) or depending on knowing the graph structure upfront. In this way, we address several key challenges of spectral-based graph neural networks simultaneously, and make our model readily applicable to inductive as well as transductive problems. Our GAT models have achieved or matched state-of-the-art results across four established transductive and inductive graph benchmarks: the Cora, Citeseer and Pubmed citation network datasets, as well as a protein-protein interaction dataset (wherein test graphs remain unseen during training).

Running with XGCN
----------------------

**Configuration template:**

.. code:: yaml

    ####### GAT-config.yaml #######

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
    model: GAT
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

    gnn_arch: "[{'in_feats': 64, 'out_feats': 64, 'num_heads': 4, 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'num_heads': 4}]"
    gnn_lr: 0.001

    loss_type: bpr
    L2_reg_weight: 0.0


**Run from CMD:**

.. code:: bash
    
    all_data_root=""       # fill your own paths here
    config_file_root=""
    
    dataset=facebook
    model=GAT
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
        --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \
        --freeze_emb 0 \
        --num_gcn_layers 2 --num_layer_sample '[10, 20]' --infer_num_layer_sample '[10, 20]' \
        --gnn_arch "[{'in_feats': 64, 'out_feats': 64, 'num_heads': 4, 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'num_heads': 4}]" \
