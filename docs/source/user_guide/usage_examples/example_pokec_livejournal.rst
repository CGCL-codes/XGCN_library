Example: Pokec and LiveJournal
=====================================

We provide data and model running scripts of two large-scale social network datasets: Pokec and LiveJournal, 
which are used in our paper: `xGCN: An Extreme Graph Convolutional Network for Large-scale Social Link Prediction <https://doi.org/10.1145/3543507.3583340>`_.
 (the Xbox dataset is an industrial one and is not able to be public.) 

All the scripts including dataset downloading & processing can be found in ``script/data_process/pokec_and_livejournal``. 
For example, to download and process the dataset, you can run the following script: 

.. code:: shell

    # fill your path here:
    all_data_root=''
    cd script/data_process/pokec_and_livejournal

    bash 00-download.sh $all_data_root
    bash 00-instance_generation.sh $all_data_root

Once the dataset instance is generated, you can run all the models. 
For example, run PPRGo and xGCN on the LiveJournal dataset (4.8 million nodes): 

.. code:: shell

    # fill your path here:
    all_data_root=''
    config_file_root='config/'
    cd script/examples/pokec_and_livejournal

    bash run_PPR-livejournal.sh  $all_data_root $config_file_root
    bash run_PPRGo-livejournal.sh  $all_data_root $config_file_root

    bash run_xGCN-livejournal.sh  $all_data_root $config_file_root

The results will be around follows: 

+-----------+-----------+------------+----------+----------------+
|           | Recall@50 | Recall@100 | NDCG@100 | Training Time  |
+===========+===========+============+==========+================+
| PPRGo     | 0.2525    | 0.3170     | 0.0950   |  20,096s       |
+-----------+-----------+------------+----------+----------------+
| xGCN      | 0.3167    | 0.3635     | 0.1349   |  5,040s        |
+-----------+-----------+------------+----------+----------------+

(Training time: time for an epoch \* number of epochs used to achieve the best score in the validation.)
