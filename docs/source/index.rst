.. XGCN documentation master file, created by
   sphinx-quickstart on Tue Feb 14 09:22:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. title:: XGCN v0.0.0

Welcome to XGCN's documentation!
===================================

XGCN is a light-weight and easy-to-use library for large-scale graph-based recommendation. 


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install


The "USER GUIDE" section is for those who want to quickly get started 
and run models on existing datasets or new datasets made by their own. 

The users only need to focus on preparing the raw dataset input 
and setting model configurations. 
We provide APIs such as ``XGCN.data.edges_split`` to split data for link prediction task. 
Users can easily run a model through the ``XGCN.main.run_model`` module. 

.. toctree::
   :maxdepth: 1
   :caption: User Guide
   
   user_guide/data_preparation
   user_guide/model_running

The "DEVELOPER GUIDE" section is for those who want to know more about 
the implementation details and create new models.

We'll first briefly introduce the overall architecture of XGCN 
and then go through the whole training process. 
Next, we'll introduce how to customize two key parts: "Models" and "DataLoaders". 

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide
   
   developer_guide/architecture_overview
   developer_guide/training_process
   developer_guide/customize_models
   developer_guide/customize_dataloaders

.. .. autosummary::
..    :toctree: generated

..    XGCN
