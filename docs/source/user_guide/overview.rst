Overview
================

The "User Guide" section is for those who want to quickly get started 
and run models on existing datasets or new datasets made by their own using XGCN. 

An overview of XGCN's running pipeline is shown in the figure below. 
XGCN receives text files as input, 
processes them into cached "dataset instances", 
and run models through CMD (command line) or APIs. 

With XGCN's high-level APIs, users only need to focus on preparing the raw dataset input 
and setting model configurations. 

.. Once the dataset instance is generated, 
.. one can easily run a model through the CMD:

.. .. code:: bash

..     python -m XGCN.main.run_model \
..         --model "GraphSAGE" \
..         --seed 1999 \
..         --data_root ... \
..         --results_root ... \
..         ...

.. Or using the APIs:

.. .. code:: python

..     config = {'model': 'xGCN', 'seed': 1999, ... }
..     # configurations parsed from command line arguments or .yaml file
    
..     data = {}
..     # a dict is needed for holding some global data objects:
    
..     # build the modules:
..     model = XGCN.build_Model(config, data)

..     train_dl = XGCN.build_DataLoader(config, data)

..     val_evaluator = XGCN.build_val_Evaluator(config, data, model)
..     test_evaluator = XGCN.build_test_Evaluator(config, data, model)

..     trainer = XGCN.build_Trainer(config, data, model, train_dl,
..                                  val_evaluator, test_evaluator)
    
..     # start training and test the model after the training process has converged
..     trainer.train_and_test()
