Overview
================

The "User Guide" section is for those who want to quickly get started 
and run models. An overview of the running pipeline is shown in the figure below: 

.. image:: ../asset/user_guide-overview.jpg
  :width: 750
  :align: center
  :alt: XGCN data process and model running pipeline

XGCN supports taking text files of the graph and evaluation sets as input,
and processing them into a standard **"dataset instance"** format 
which is used by all the downstream models. 
Once a dataset instance is generated, you can easily run models with XGCN's APIs. 

In the following, we'll introduce how to use XGCN from the four aspects below:

* :ref:`Data Preparation <user_guide-data_preparation>`: how to process your text data into our "dataset instances". 

* :ref:`Training and Evaluation <user_guide-training_and_evaluation>`: the model training and evaluation APIs. 

* :ref:`Supported Models <user_guide-supported_models>`: available models and their configurations. 

* :ref:`Usage Examples <user_guide-usage_examples>`: end-to-end usage examples. 
