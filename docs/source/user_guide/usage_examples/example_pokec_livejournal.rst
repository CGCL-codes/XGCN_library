Example: Pokec and LiveJournal
=====================================

We provide data and model running scripts of two large-scale social network datasets: Pokec and LiveJournal, 
which are used in our xGCN paper (the Xbox dataset is an industrial one and is not able to be public.) 

The data can be downloaded from here: 
`pokec_and_livejournal_data <https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip>`_. 
To download and process them, please refer to the "Data Preparation" section and 
the scripts in ``script/data_process/pokec_and_livejournal``. 

Please refer to ``script/examples/pokec_and_livejournal`` which includes all the scripts for different datasets. 
To run a model, you only need to modify the ``all_data_root`` and ``config_file_root`` 
arguments in the script to your own paths. 
