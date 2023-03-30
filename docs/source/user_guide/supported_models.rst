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
