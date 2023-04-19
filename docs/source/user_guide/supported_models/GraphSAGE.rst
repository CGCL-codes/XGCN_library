.. _supported_models-GraphSAGE:

GraphSAGE
===================

-----------------
Introduction
-----------------

`\[paper\] <https://arxiv.org/abs/1706.02216>`_

**Title:** Inductive Representation Learning on Large Graphs

**Authors:** William L. Hamilton, Rex Ying, Jure Leskovec

**Abstract:** Low-dimensional embeddings of nodes in large graphs have proved extremely useful in a variety of prediction tasks, from content recommendation to identifying protein functions. However, most existing approaches require that all nodes in the graph are present during training of the embeddings; these previous approaches are inherently transductive and do not naturally generalize to unseen nodes. Here we present GraphSAGE, a general, inductive framework that leverages node feature information (e.g., text attributes) to efficiently generate node embeddings for previously unseen data. Instead of training individual embeddings for each node, we learn a function that generates embeddings by sampling and aggregating features from a node's local neighborhood. Our algorithm outperforms strong baselines on three inductive node-classification benchmarks: we classify the category of unseen nodes in evolving information graphs based on citation and Reddit post data, and we show that our algorithm generalizes to completely unseen graphs using a multi-graph dataset of protein-protein interactions.

----------------------
Running with XGCN
----------------------

forward_mode: 'full_graph'
--------------------------

When using the 'full_graph' forward_mode, embeddings of all the nodes are inferred in each 
training batch. 

**Configuration template:**

.. code:: yaml

    # config/GraphSAGE-full_graph-config.yaml
    # Dataset/Results root
    data_root: ""
    results_root: ""

    # Trainer configuration
    epochs: 200
    use_validation_for_early_stop: 1
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20
    val_method: ""
    val_batch_size: 256
    file_val_set: ""

    # Testing configuration (optional)
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

    # DataLoader configuration
    Dataset_type: NodeListDataset
    num_workers: 0
    NodeListDataset_type: LinkDataset
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    str_num_total_samples: num_edges
    epoch_sample_ratio: 0.1

    # Model configuration
    model: GraphSAGE
    seed: 1999

    graph_device: "cuda:0"
    emb_table_device: "cuda:0"
    gnn_device: "cuda:0"
    out_emb_table_device: "cuda:0"

    forward_mode: full_graph

    from_pretrained: 0
    file_pretrained_emb: ""
    freeze_emb: 0
    use_sparse: 0
    emb_dim: 64 
    emb_init_std: 0.1
    emb_lr: 0.005

    gnn_arch: "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]"
    gnn_lr: 0.01

    loss_type: bpr
    L2_reg_weight: 0.0

**Run from command line:**

.. code:: bash
    
    # script/examples/facebook/run_GraphSAGE-full_graph.sh
    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=facebook
    model=GraphSAGE
    seed=0
    device="cuda:1"
    graph_device=$device
    emb_table_device=$device
    gnn_device=$device
    out_emb_table_device=$device

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    # file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-full_graph-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method one_pos_k_neg \
        --file_val_set $data_root/val-one_pos_k_neg.pkl \
        --key_score_metric r20 \
        --test_method multi_pos_whole_graph \
        --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
        --graph_device $graph_device --emb_table_device $emb_table_device \
        --gnn_device $gnn_device --out_emb_table_device $out_emb_table_device \
        # --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \


forward_mode: 'sample'
--------------------------

When using the 'sample' forward_mode, DGL's neighbor sampler is used to generate "blocks" 
(please refer to `DGL docs: Chapter 6: Stochastic Training on Large Graphs <https://docs.dgl.ai/en/latest/guide/minibatch.html>`_ for more information). 


**Configuration template:**

.. code:: yaml

    # config/GraphSAGE-block-config.yaml
    # Dataset/Results root
    data_root: ""
    results_root: ""

    # Trainer configuration
    epochs: 200
    use_validation_for_early_stop: 1
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20
    val_method: ""
    val_batch_size: 256
    file_val_set: ""

    # Testing configuration (optional)
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

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
    str_num_total_samples: num_edges
    epoch_sample_ratio: 0.1

    # Model configuration
    model: GraphSAGE
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

    gnn_arch: "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]"
    gnn_lr: 0.01

    loss_type: bpr
    L2_reg_weight: 0.0


**Run from command line:**

.. code:: bash
    
    # script/examples/facebook/run_GraphSAGE-block.sh
    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=facebook
    model=GraphSAGE
    seed=0
    device="cuda:1"
    graph_device=$device
    emb_table_device=$device
    gnn_device=$device
    out_emb_table_device=$device

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    # file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-block-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method one_pos_k_neg \
        --file_val_set $data_root/val-one_pos_k_neg.pkl \
        --key_score_metric r20 \
        --test_method multi_pos_whole_graph \
        --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
        --graph_device $graph_device --emb_table_device $emb_table_device \
        --gnn_device $gnn_device --out_emb_table_device $out_emb_table_device \
        # --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \
