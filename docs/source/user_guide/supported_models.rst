.. _user_guide-supported_models:

Supported Models
====================

XGCN now supports a range of classic GNN models such as GraphSAGE and GAT, 
and several recent models for recommendation (collaborative filtering) such as LightGCN and UltraGCN. 
XGCN also includes **xGCN** - an implementation for the TheWebConf 2023 paper: 
**xGCN: An Extreme Graph Convolutional Network for Large-scale Social Link Prediction**. 

From the view of scaling strategy, XGCN supports three kinds of common methods, which can be 
mounted on different GNN neural architectures: 

* Layer-sampling (sampling neighbors in each GNN layer, which is proposed in GraphSAGE). 

* Cluster-sampling (sampling graph clusters, which is proposed in Cluster-GCN). 

* Graph-reshaping (cancelling multi-layer message-passing, such as PPRGo). 

We provide running examples for each model and each scaling strategy, 
the full model list is as follows: 

.. toctree::
    :maxdepth: 1

    supported_models/GAMLP.rst
    supported_models/GAT.rst
    supported_models/GBP.rst
    supported_models/GIN.rst
    supported_models/GraphSAGE.rst
    supported_models/LightGCN.rst
    supported_models/Node2vec.rst
    supported_models/PPRGo.rst
    supported_models/SGC.rst
    supported_models/SIGN.rst
    supported_models/SimpleX.rst
    supported_models/SSGC.rst
    supported_models/UltraGCN.rst
    supported_models/xGCN.rst
