.. XGCN documentation master file, created by
   sphinx-quickstart on Tue Feb 14 09:22:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. title:: XGCN v0.0.0

Welcome to XGCN's documentation!
===================================

XGCN is a light-weight and easy-to-use library for large-scale graph-based recommendation, 
aiming at helping researchers to build models that can run on million-scale graphs. 
We provides an implementation for the TheWebConf 2023 paper: 
**xGCN: An Extreme Graph Convolutional Network for Large-scale Social Link Prediction**. 

Features:

- Large-scale GNN training
   XGCN targets at presenting optimized GNN recommendation models with scaling strategies that 
   can easily run on million-scale graphs. 
   We include xGCN - a brand new GNN model which can embed a 100-million graph 
   within 1 day under a single-machine environment. 
- A complete data pipeline for large graphs
   XGCN covers a complete machine learning pipeline: from dataset making to model evaluation. 
   We provide tools to efficiently process large graphs in CSR format. 
- Easy-to-use infrastructure
   XGCN is friendly to those who want to create new models. 
   We provide clear interface for each module. One can easily develop a new model 
   by inheriting a base class such as ``BaseEmbeddingModel``. 


.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/introduction
   user_guide/data_preparation
   user_guide/model_running

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/introduction
   developer_guide/Model
   developer_guide/DataLoader
   developer_guide/Evaluator
   developer_guide/Trainer
