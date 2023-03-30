Running Examples
===================

We provide model running examples on two large-scale social network dataset: Pokec and LiveJournal, 
which are used in our xGCN paper.

The data can be downloaded from here: 
`pokec_and_livejournal_data <https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip>`_. 
(The Xbox dataset is an industrial one and is not able to be public.) 
To download and process them, please refer to the "Data Preparation" section and 
the scripts in ``script/data_process/pokec_and_livejournal``. 

Please refer to ``script/model`` which includes all the scripts for different 
datasets (The used datasets in our paper are: Pokec, LiveJournal, and Xbox. 
Note that we do not tune models on the facebook dataset example): 

.. code:: 

    script
    └── model
        ├─ GAMLP
        ├─ ...
        └─ xGCN
           ├─ run_xGCN-facebook.sh
           ├─ run_xGCN-livejournal.sh
           ├─ run_xGCN-pokec.sh
           └─ run_xGCN-xbox-3m.sh

To run a model, you only need to modify the ``all_data_root`` and ``config_file_root`` 
arguments in the script to your own paths. 
