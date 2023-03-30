Node2vec
=============

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
