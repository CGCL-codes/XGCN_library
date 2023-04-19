.. _supported_models-GIN:

GIN
=======

-----------------
Introduction
-----------------

`\[paper\] <https://arxiv.org/abs/1810.00826>`_

**Title:** How Powerful are Graph Neural Networks?

**Authors:** Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka

**Abstract:** Graph Neural Networks (GNNs) are an effective framework for representation learning of graphs. GNNs follow a neighborhood aggregation scheme, where the representation vector of a node is computed by recursively aggregating and transforming representation vectors of its neighboring nodes. Many GNN variants have been proposed and have achieved state-of-the-art results on both node and graph classification tasks. However, despite GNNs revolutionizing graph representation learning, there is limited understanding of their representational properties and limitations. Here, we present a theoretical framework for analyzing the expressive power of GNNs to capture different graph structures. Our results characterize the discriminative power of popular GNN variants, such as Graph Convolutional Networks and GraphSAGE, and show that they cannot learn to distinguish certain simple graph structures. We then develop a simple architecture that is provably the most expressive among the class of GNNs and is as powerful as the Weisfeiler-Lehman graph isomorphism test. We empirically validate our theoretical findings on a number of graph classification benchmarks, and demonstrate that our model achieves state-of-the-art performance.

----------------------
Running with XGCN
----------------------

forward_mode: 'full_graph'
--------------------------

When using the 'full_graph' forward_mode, embeddings of all the nodes are inferred in each 
training batch. 

**Configuration template:**

.. code:: yaml

    # config/GIN-full_graph-config.yaml
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

    # Testing configuration
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
    model: GIN
    seed: 1999

    graph_device: "cuda:0"
    emb_table_device: "cuda:0"
    gnn_device: "cuda:0"
    out_emb_table_device: "cuda:0"

    forward_mode: full_graph
    num_gcn_layers: 2

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

**Run from command line:**

.. code:: bash
    
    # script/examples/facebook/run_GIN-full_graph.sh
    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=facebook
    model=GIN
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

    # config/GIN-block-config.yaml
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

    # Testing configuration
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


**Run from command line:**

.. code:: bash
    
    # script/examples/facebook/run_GIN-block.sh
    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=facebook
    model=GIN
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
