Introduction
===============

The "Developer Guide" section is for those who want to know more about 
the implementation details and develop new models. 

XGCN has four key modules: ``Model``, ``DataLoader``, ``Evaluator``, and ``Trainer``.  
An overview of their interactions is shown in the figure below:

.. image:: ../asset/overview.jpg
  :width: 600
  :alt: key modules (Model, DataLoader, Evaluator, and Trainer) and their interactions

``Trainer`` lies in the center of the control flow and is responsible for 
the whole model training process. 
``Model`` is in the center of the data flow and receives training/evaluation data. 
``DataLoader`` feeds batch training data to the ``Model``. 
``Evaluator`` sends batch evaluation data to the ``Model``, receives inference outputs, 
and calculates accuracy metrics. 

To initialize these modules and run a model, 
one can use the high-level APIs such as ``XGCN.build_Model()``, as shown below: 

.. code:: python

    # XGCN/main/run_model.py

    config = {'model': 'xGCN', 'seed': 1999, ... }
    # configurations parsed from command line arguments or .yaml file
    
    data = {}
    # a dict is needed for holding some global data objects:
    
    # build the modules:
    model = XGCN.build_Model(config, data)

    train_dl = XGCN.build_DataLoader(config, data)

    val_evaluator = XGCN.build_val_Evaluator(config, data, model)
    test_evaluator = XGCN.build_test_Evaluator(config, data, model)

    trainer = XGCN.build_Trainer(config, data, model, train_dl,
                                 val_evaluator, test_evaluator)
    
    # start training and test the model after the training process has converged
    trainer.train_and_test()

For more information, you can referance ``XGCN/main/run_model.py``.
