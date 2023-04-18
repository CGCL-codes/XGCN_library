.. _user_guide-training_and_evaluation-model_evaluation:

Model Evaluation
======================

To test a model, you can just call ``model.test()``, 
it executes the default testing setting configurations in ``config``: 

.. code:: python

    config = {
        'data_root': ..., 'results_root': ..., 
        'model': 'xGCN', 'seed': 1999, 
        ...,
        'test_method': 'OnePosKNeg_Evaluator', 
        'test_batch_size': 256, 'file_test_set': ...,
        ...
    }
    model = XGCN.create_model(config)
    model.fit()  
    results = model.test()

Or you can specify other test sets:

.. code:: python

    test_config = {
        'test_method': 'multi_pos_whole_graph',
        'test_batch_size': 256,
        'file_test_set': ... 
    }
    results = model.test(test_config)

The function receives a Dict containing three arguments: 

* ``test_method``: specifices the evaluation method. Available options: 'one_pos_k_neg', 'one_pos_whole_graph', and 'multi_pos_whole_graph'. 

* ``test_batch_size``: specifices the batch size. 

* ``file_test_set``: specifices the file of the processed evaluation set. 