Data Preparation
======================

XGCN supports handling both homogenous graphs (e.g. social networks) 
and bipartite graphs (e.g. user-item graphs). 
The processing pipeline is as follows:

**(to add figure)**

As shown in the figure, XGCN has three data process modules: 

* ``XGCN.data.io``. Handle disk-reading/writing operations, including 
reading raw token IDs as input and re-index them into continuous integer IDs. 

* ``XGCN.data.trim``. Some real-world graph datasets tend to be large-scale 
(e.g. **(to add an exapmle)**), and we might want to get a smaller one for model 
development and quick evaluation. This module help users trim graphs by dropping 
nodes randomly or according to nodes' degrees. 

* ``XGCN.data.split_edges``. To evaluate a link prediction model, it is common to 
split a portion of edges as positive samples, which can be done by using this module. 

In this section, we'll first introduce the "dataset instance" format, which is needed 
to run a model, and then we'll provide data processing examples on two small datasets: 

* **facebook**. The facebook data is included in our XGCN repository: 
``data/raw_facebook/``. You can also download it from SNAP: 
`facebook_combined.txt.gz <http://snap.stanford.edu/data/facebook_combined.txt.gz>`_. 

* **gowalla**. The gowallla data is also included here: ``data/raw_gowalla``. You can also 
found it in LightGCN's official repository: `<https://github.com/gusye1234/LightGCN-PyTorch>`_.

We provide the Poke and LiveJournal datasets used in the xGCN paper: 
`pokec_and_livejournal_data <https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip>`_. 
To download and process them, please refer to the following introduction sections 
and the scripts in ``script/data_process/pokec_and_livejournal``. 

.. toctree::
    :maxdepth: 1
   
    data_preparation/dataset_instance
    data_preparation/exapmle_facebook
    data_preparation/exapmle_gowalla
    data_preparation/data_process_api
