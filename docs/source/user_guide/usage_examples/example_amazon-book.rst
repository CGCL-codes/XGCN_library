Example: amazon-book
======================

Here we present a user-item graph example. 
We run LightGCN and xGCN on the amazon-book dataset, which is used in the LightGCN paper. 
The results are as follows: 

+-----------+-----------+----------+----------------+
|           | Recall@20 | NDCG@20  | Training Time  |
+===========+===========+==========+================+
| LightGCN  | 0.0409    | 0.0316   |  69,954s       |
+-----------+-----------+----------+----------------+
| xGCN      | 0.0452    | 0.0355   |  8,135s        |
+-----------+-----------+----------+----------------+

(Training time: time for an epoch \* number of epochs used to achieve the best score in the validation.)

The amazon-book dataset can be found 
in our XGCN repository: ``data/raw_amazon-book/``, which is copied from LightGCN's official code repository: 
https://github.com/gusye1234/LightGCN-PyTorch.


---------------------
Data Preparation
---------------------

Before getting started
-------------------------

We recommend to arrange the data with a clear directory structure. 
Before getting started, you may manually 
setup an ``XGCN_data`` (or other names you like) directory as follows: 
(It's recommended to put your ``XGCN_data`` somewhere else than in this repository.)

.. code:: 

    XGCN_data
    └── dataset
        └── raw_amazon-book
            ├── train.txt
            └── test.txt

We'll use this directory to hold all the different datasets 
and models outputs. 
We refer to its path as ``all_data_root`` in our scripts. 


Dataset instance generation
-----------------------------

First, let's process the graph: 

.. code:: shell

    ###### process graph for training
    # set to your own path:
    file_input_graph='/home/xxx/XGCN_data/dataset/raw_amazon-book/train.txt'
    data_root='/home/xxx/XGCN_data/dataset/instance_amazon-book'
    
    mkdir -p $data_root  # make sure to setup the directory

    graph_type='homo'
    graph_format='edge_list'

    python -m XGCN.data.process.process_int_graph \
        --file_input_graph $file_input_graph --data_root $data_root \
        --graph_type $graph_type --graph_format $graph_format \

Then we process the test set (the LightGCN paper does not provide a validation set): 

.. code:: shell

    ###### process test set
    file_input='/home/xxx/XGCN_data/dataset/raw_amazon-book/test.txt'
    file_output='/home/xxx/XGCN_data/dataset/instance_amazon-book/test.pkl'

    evaluation_method='multi_pos_whole_graph'

    python -m XGCN.data.process.process_evaluation_set \
        --file_input $file_input --file_output $file_output \
        --evaluation_method $evaluation_method \

The test set is large (52643 source nodes) and the testing process is time-consuming. 
So here we sample some source nodes for quick validation during the model training. 
Here is the python script: 

.. code:: python

    # script/examples/amazon-book/sample_from_test_set_for_validation.py
    from XGCN.data import io
    from XGCN.utils.parse_arguments import parse_arguments

    import numpy as np


    def main():
        
        config = parse_arguments()
        file_input = config['file_input']
        file_output = config['file_output']
        num_sample = config['num_sample']
        
        test_set = io.load_pickle(file_input)
        src = test_set['src']
        pos_list = test_set['pos_list']
        print("number of souce node in the test set:", len(src))
        print("num_sample:", num_sample)
        
        np.random.seed(1999)
        idx = np.arange(len(src))
        np.random.shuffle(idx)
        sampled_idx = idx[:num_sample]
        
        val_src = src[sampled_idx]
        val_pos_list = []
        pos_list = test_set['pos_list']
        for i in sampled_idx:
            val_pos_list.append(pos_list[i])

        val_set = {
            'src': val_src,
            'pos_list': val_pos_list
        }
        io.save_pickle(file_output, val_set)


    if __name__ == '__main__':
        
        main()


Here is the corresponding shell script: 

.. code:: shell

    ###### sample from the test set
    python sample_from_test_set_for_validation.py \
        --file_input $all_data_root"/dataset/instance_amazon-book/test.pkl" \
        --file_output $all_data_root"/dataset/instance_amazon-book/val.pkl" \
        --num_sample 3000 \

After the above processing, your data directory will look like this: 

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_amazon-book
        |   ├── train.txt
        |   └── test.txt
        └── instance_amazon-book
            ├── info.yaml
            ├── indices.pkl
            ├── indptr.pkl
            ├── val.pkl
            └── test.pkl

The whole processing script can be found in ``script/examples/amazon-book/00-instance_generation.sh``. 

-----------------
Run LightGCN
-----------------

XGCN provides a simple module - ``XGCN.main.run_model`` - to run models from command line. 
It has the following contents:

.. code:: python

    import XGCN
    from XGCN.data import io
    from XGCN.utils.parse_arguments import parse_arguments

    import os.path as osp


    def main():
        
        config = parse_arguments()

        model = XGCN.create_model(config)
        
        model.fit()
        
        test_results = model.test()
        print("test:", test_results)
        io.save_json(osp.join(config['results_root'], 'test_results.json'), test_results)


    if __name__ == '__main__':
        
        main()


The following shell script runs LightGCN with ``XGCN.main.run_model`` module and 
reproduce the results on the amazon-book dataset: 

.. code:: shell

    # script/examples/amazon-book/01-run_LightGCN.sh
    # The results of the following running should be around:
    # r20:0.0409 || r50:0.0792 || r100:0.1252 || r300:0.2367 || n20:0.0316 || n50:0.0458 || n100:0.0606 || n300:0.0911
    # 'r' for 'Recall@', 'n' for 'NDCG@'

    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=amazon-book
    model=LightGCN
    seed=0
    device="cuda:0"
    graph_device=$device
    emb_table_device=$device
    gnn_device=$device
    out_emb_table_device=$device

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    # In LightGCN's official code (https://github.com/gusye1234/LightGCN-PyTorch), 
    # for each epoch, there are num_edges samples. For each sample, firstly, a user 
    # is randomly sampled. Then a neighbor (item) of the user is sampled as the positive node. 

    # The amazon-book dataset has 52643 users and 2380730 interactions (edges). 
    # 2380730 / 52643 = 45.22
    # To reproduce the LightGCN's setting, in XGCN, we use the 
    # NodeBased_ObservedEdges_Sampler, and set:
    # str_num_total_samples=num_users
    # epoch_sample_ratio=45.22

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-full_graph-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method multi_pos_whole_graph \
        --file_val_set $data_root/test.pkl \
        --test_method multi_pos_whole_graph \
        --file_test_set $data_root/test.pkl \
        --str_num_total_samples num_users \
        --pos_sampler NodeBased_ObservedEdges_Sampler \
        --neg_sampler StrictNeg_Sampler \
        --epoch_sample_ratio 45.22 \
        --num_gcn_layers 2 \
        --L2_reg_weight 1e-4 --use_ego_emb_L2_reg 1 \
        --emb_lr 0.001 \
        --emb_dim 64 \
        --train_batch_size 2048 \
        --epochs 1000 --val_freq 20 \
        --key_score_metric r20 --convergence_threshold 100 \
        --graph_device $graph_device --emb_table_device $emb_table_device \
        --gnn_device $gnn_device --out_emb_table_device $out_emb_table_device \

-----------------
Run xGCN
-----------------

The following shell script runs xGCN with ``XGCN.main.run_model``: 

.. code:: shell

    # script/examples/amazon-book/01-run_xGCN.sh
    # The results of the following running should be around:
    # r20:0.0452 || r50:0.0844 || r100:0.1302 || r300:0.2398 || n20:0.0355 || n50:0.0501 || n100:0.0650 || n300:0.0951
    # 'r' for 'Recall@', 'n' for 'NDCG@'

    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=amazon-book
    model=xGCN
    seed=0
    device='cuda:0'
    emb_table_device=$device
    forward_device=$device
    out_emb_table_device=$device

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed][epoch_sample_ratio1.0]

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method multi_pos_whole_graph \
        --file_val_set $data_root/val.pkl \
        --test_method multi_pos_whole_graph \
        --file_test_set $data_root/test.pkl \
        --emb_table_device $emb_table_device \
        --forward_device $forward_device \
        --out_emb_table_device $out_emb_table_device \
        --epochs 1000 --val_freq 1 --convergence_threshold 100 \
        --key_score_metric r20 \
        --epoch_sample_ratio 1.0 \
        --dnn_arch "[nn.Linear(64, 1024), nn.Tanh(), nn.Linear(1024, 64)]" \
        --use_scale_net 0 \
        --L2_reg_weight 1e-4 \
        --num_gcn_layers 1 \
        --stack_layers 1 \
        --renew_by_loading_best 1 \
        --T 5 \
        --K 99999 \
        --tolerance 5 \

-----------------------
The Complete Scripts
-----------------------

All the scripts of this running example can be found in ``script/examples/amazon-book``. 
Remember to modify ``all_data_root`` and ``config_file_root`` in the shell scripts to your own paths. 
After the raw data preparation, you can run all the code by:

.. code:: bash

    cd script/examples/amazon-book
    bash 00-instance_generation.sh
    bash 01-run_LightGCN.sh
    bash 02-run_xGCN.sh
