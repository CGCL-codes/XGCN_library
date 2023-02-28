Model Training
=========================

Users can easily run a model with the ``gnn_zoo.main.run_model`` module: 

.. code:: bash

    python -m gnn_zoo.main.run_model \
        --model "GraphSAGE" \
        --seed 1999 \
        --data_root /xxx/xxx \
        --results_root /xxx/xxx \

In this section, we introduce the configuration format, the setting of model input and 

1. Configuration Format
-----------------------------

GNN_ZOO supports parsing model configurations from command line arguments and ``.yaml`` files. 
Directory ``scripts/`` provides examples of ``.sh`` shell scripts to run all the models. 

If you want to use a ``.yaml`` configuration file, specify the path 
with the command line argument ``--config_file`` like follows:

.. code:: bash

    python -m gnn_zoo.main.run_model \
        --config_file "../config/GraphSAGE/config.yaml" \
        --seed 1999 \
        ...

Directory ``config/`` contains ``.yaml`` configuration files which 
include all the arguments needed to run the models. 
A ``.yaml`` file is not a necessity of running the code and has lower 
priority when the same command line argument is given. 


2. Training Data and Results
-----------------------------


In the last section, we process the raw facebook data and generate a dataset instance:

.. code:: 

    gnn_zoo_data
    └── dataset
        └── instance_facebook
            ├── indices.pkl
            ├── indptr.pkl
            ├── info.yaml
            ├── pos_edges.pkl
            ├── test_set.pkl
            └── val_set.pkl

With these cached data, we can run all the models by specifying the ``data_root`` in the configuration, 
which is ``/xxx/gnn_zoo_data/dataset/instance_facebook`` here. 
We use the
