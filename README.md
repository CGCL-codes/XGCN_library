# XGCN

XGCN is a light-weight and easy-to-use library for large-scale graph-based recommendation, 
aiming at helping researchers to build models that can run on million-scale graphs. 

XGCN includes xGCN - an implementation for the TheWebConf 2023 paper: 
**xGCN: An Extreme Graph Convolutional Network for Large-scale Social Link Prediction**. 

For more information, Please refer to our documentations: [Docs](https://xgcn.readthedocs.io/en/latest/index.html)

## Features
+ **Large-scale GNN training.**
   XGCN targets at presenting optimized GNN recommendation models with scaling strategies that 
   can easily run on million-scale graphs. 
   We include xGCN - a brand new GNN model which can quickly embed large graphs.
+ **A complete data pipeline for large graphs.**
   XGCN covers a complete machine learning pipeline: from dataset making to model evaluation. 
   We provide tools to efficiently process large graphs in CSR format. 
+ **Easy-to-use infrastructure.**
   XGCN is friendly to those who want to create new models. 
   We provide clear interface for each module. One can easily develop a new model 
   by inheriting a base class such as ``BaseEmbeddingModel``. 
