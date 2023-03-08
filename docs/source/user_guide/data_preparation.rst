Data Preparation
======================

XGCN supports handling both social graphs (all the nodes are users) and user-item graphs. 
In this section, let's take a small dataset - `facebook <http://snap.stanford.edu/data/ego-Facebook.html>`_ - 
as an example, start from the raw ``.txt`` file of a graph 
and go through the whole data preparation pipeline. 

The facebook data is included in our XGCN repository: ``data/raw_facebook/facebook_combined.txt``. 
You can also download it from SNAP: `facebook_combined.txt.gz <http://snap.stanford.edu/data/facebook_combined.txt.gz>`_. 

We also provide the Poke and LiveJournal datasets used in the xGCN paper: 
`pokec_and_livejournal_data <https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip>`_. 
To download and process them, please refer to the following introduction sections 
and the scripts in ``script/data_process/pokec_and_livejournal``. 

.. toctree::
    :maxdepth: 1
   
    data_preparation/dataset_instance
    data_preparation/prepare_raw_data
    data_preparation/instance_making
