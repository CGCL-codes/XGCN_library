Node2vec
=============

Introduction
-----------------

`\[paper\] <https://dl.acm.org/doi/10.1145/2939672.2939754>`_

**Title:** node2vec: Scalable Feature Learning for Networks

**Authors:** Aditya Grover, Jure Leskovec

**Abstract:** Prediction tasks over nodes and edges in networks require careful effort in engineering features used by learning algorithms. Recent research in the broader field of representation learning has led to significant progress in automating prediction by learning the features themselves. However, present feature learning approaches are not expressive enough to capture the diversity of connectivity patterns observed in networks. Here we propose node2vec, an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes. We define a flexible notion of a node's network neighborhood and design a biased random walk procedure, which efficiently explores diverse neighborhoods. Our algorithm generalizes prior work which is based on rigid notions of network neighborhoods, and we argue that the added flexibility in exploring neighborhoods is the key to learning richer representations. We demonstrate the efficacy of node2vec over existing state-of-the-art techniques on multi-label classification and link prediction in several real-world networks from diverse domains. Taken together, our work represents a new way for efficiently learning state-of-the-art task-independent representations in complex networks.

Running with XGCN
----------------------

**Configuration template for Node2vec:**

.. code:: yaml

    # Dataset/Results root
    data_root: ""
    results_root: ""

    # DataLoader configuration
    train_batch_size: 2048
    num_workers: 4

    # Trainer configuration
    epochs: 200
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20

    # Evaluator configuration
    val_evaluator: WholeGraph_MultiPos_Evaluator
    val_batch_size: 256
    file_val_set: ""
    test_evaluator: WholeGraph_MultiPos_Evaluator
    test_batch_size: 256
    file_test_set: ""

    # Model configuration
    model: Node2vec
    seed: 1999
    device: cpu
    emb_dim: 64
    emb_lr: 0.005
    walk_length: 12
    num_walks: 8
    context_size: 5
    p: 1.0
    q: 1.0
    num_neg: 4


**Run Node2vec from CMD:**

.. code:: bash

    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset=facebook
    model=Node2vec
    seed=0
    device="cuda:0"

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_evaluator WholeGraph_MultiPos_Evaluator --val_batch_size 256 \
        --file_val_set $data_root/val_set.pkl \
        --test_evaluator WholeGraph_MultiPos_Evaluator --test_batch_size 256 \
        --file_test_set $data_root/test_set.pkl \
        --device $device \


**Configuration template for GensimNode2vec:**

.. code:: yaml

    seed: 1999
    model: GensimNode2vec

    data_root: ""
    results_root: ""

    val_evaluator: ""
    val_batch_size: 256
    file_val_set: ""

    test_evaluator: ""
    test_batch_size: 256
    file_test_set: ""

    epochs: 200
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20

    emb_dim: 64
    emb_lr: 0.01
    num_walks: 16
    walk_length: 16
    p: 1.0
    q: 1.0
    context_size: 5
    num_neg: 5

    num_workers: 6


**Run GensimNode2vec from CMD:**

.. code:: bash

    all_data_root=""       # fill your own paths here
    config_file_root=""

    dataset="facebook"
    model="GensimNode2vec"

    data_root=$all_data_root"/dataset/instance_"$dataset

    seed=1999

    results_root=$all_data_root"/model_output/"$dataset"/"$model"/[seed$seed]"

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_evaluator "WholeGraph_MultiPos_Evaluator" --val_batch_size 256 \
        --file_val_set $data_root"/val_set.pkl" \
        --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
        --file_test_set $data_root"/test_set.pkl" \
        --epochs 200 --val_freq 1 \
        --emb_dim 64 --emb_lr 0.01 \
        --num_walks 16 --walk_length 16 \
        --p 1.0 --q 1.0 \
        --context_size 5 --num_neg 5 \
        --num_workers 6 \
