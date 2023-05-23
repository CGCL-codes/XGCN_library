.. _user_guide-training_and_evaluation:

Training and Evaluation
============================

Once the dataset instance is generated (for dataset instance generation, please refer to the previous section: :ref:`Data Preparation <user_guide-data_preparation>`), 
you can run models with XGCN's APIs: 

.. code:: python

    model = XGCN.create_model(config)                  # create model
    model.fit()                                        # train
    score, topk_node = model.infer_topk(k=100, src=5)  # inference

In this section, we are going to introduce:

.. toctree::
   :maxdepth: 1

   training_and_evaluation/model_configuration
   training_and_evaluation/model_training
   training_and_evaluation/model_evaluation
   training_and_evaluation/model_inference
