Overview
================

The "User Guide" section is for those who want to quickly get started 
and run models on existing datasets or new datasets made by their own using XGCN. 

An overview of the running pipeline is shown in the figure below. 
XGCN supports process raw text files or DGLGraph and 
specifiy a "dataset instance" format which is used by the downstream models. 
Once a dataset instance is generated, you can run models with CMD or API functions. 

.. image:: ../asset/user_guide-overview.jpg
  :width: 500
  :align: center
  :alt: XGCN data process and model running pipeline

In the following, we'll introduce how to use XGCN from three aspects:

* Data Preparation

* Model Training

* Model Evaluation

.. .. image:: ../asset/xgcn_fig1.jpg
..   :width: 500
..   :align: center
..   :alt: xGCN efficiency study

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
