.. XGCN documentation master file, created by
   sphinx-quickstart on Tue Feb 14 09:22:13 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
.. title:: XGCN v0.0.0

Welcome to XGCN's documentation!
===================================

XGCN is a light-weight and easy-to-use library for large-scale Graph Neural Network (GNN) embedding, 
aiming at helping researchers to quickly embed million-scale graphs in a single-machine environment. 
XGCN includes xGCN - an implementation for the TheWebConf 2023 paper: 
**xGCN: An Extreme Graph Convolutional Network for Large-scale Social Link Prediction**. 
Our repository is available at: https://github.com/xiransong/xGCN .

**Features**:

- xGCN - a brand new GNN embedding model
   XGCN present an implementation of xGCN, which achieves best accuracy and efficiency over 
   recent existing models on large graphs, including an industrial dataset with 100 million nodes. 

.. image:: ../asset/xgcn_fig1.png
  :width: 600
  :alt: xGCN efficiency study

.. image:: ../asset/xgcn_fig2.png
  :width: 600
  :alt: xGCN on Xbox-100m dataset

- Large-scale GNN embedding
   XGCN targets at presenting GNN embedding models that can easily run on million-scale graphs 
   in a single-machine environment. 
   Some official implementations (e.g. LightGCN, UltraGCN) 
   mainly consider small datasets and do not scale to large graphs.  
   We fully utilize DGL and PyTorch's mechanism and present models that scale to large graphs. 

- A complete data pipeline for large graphs
   XGCN covers a complete machine learning pipeline: from dataset making to model evaluation. 
   We provide tools to efficiently process large graphs in CSR format. 

- Easy-to-use infrastructure
   XGCN is friendly to those who want to create new models. 
   We provide clear interface for each module. One can easily develop a new model 
   by inheriting a base class such as ``BaseEmbeddingModel``. 


Install
------------------

We recommend to install XGCN from source with the following command:
(Python \>= 3.8, torch \>= 1.7.0, dgl \>= 0.9, torch_geometric \>= 2.0 are required.)

.. code:: bash

    git clone git@github.com:xiransong/xGCN.git -b XGCN_dev
    cd xGCN
    python -m pip install -e .
   

Full Documentations
------------------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   index
   get_started/install

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/introduction
   user_guide/data_preparation
   user_guide/model_running
   user_guide/reproduction

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/introduction
   developer_guide/Model
   developer_guide/DataLoader
