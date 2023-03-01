Architecture Overview
=========================


The code architecture of XGCN is summarized as the figure below. 
As shown in the figure, there are four key moduels: 
``Trainer``, ``DataLoader``, ``Model``, and ``Evaluator``.

.. image:: ../asset/overview.jpg
  :width: 600
  :alt: overview of the code architecture

The ``Trainer`` lies in the center of control flow and is responsible for 
the whole model training process. 
The ``Model`` is in the center of data flow and receives training/evaluation data. 
The ``DataLoader`` feeds batch training data to the ``Model``. 
The ``Evaluator`` sends batch evaluation data to the ``Model``, receives inference outputs, 
and calculates accuracy metrics.

To construct these modules and run a model, 
one can use the high-level APIs such as ``XGCN.build_Model``, as shown below: 

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
