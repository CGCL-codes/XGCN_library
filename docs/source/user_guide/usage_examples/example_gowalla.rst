Example: gowalla
======================

Here we present a user-item graph example. The used gowalla dataset can be found 
in our XGCN repository: ``data/raw_gowalla/``, which is copied from LightGCN's official code repository: 
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
        └── raw_gowalla
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
    file_input_graph='/home/xxx/XGCN_data/dataset/raw_gowalla/train.txt'
    data_root='/home/xxx/XGCN_data/dataset/instance_gowalla'
    
    mkdir -p $data_root  # make sure to setup the directory

    graph_type='homo'
    graph_format='edge_list'

    python -m XGCN.data.process.process_int_graph \
        --file_input_graph $file_input_graph --data_root $data_root \
        --graph_type $graph_type --graph_format $graph_format \

Then we process the test set (the LightGCN paper does not provide a validation set): 

.. code:: shell

    ###### process test set
    file_input='/home/xxx/XGCN_data/dataset/raw_gowalla/test.txt'
    file_output='/home/xxx/XGCN_data/dataset/instance_gowalla/test.pkl'

    evaluation_method='multi_pos_whole_graph'

    python -m XGCN.data.process.process_evaluation_set \
        --file_input $file_input --file_output $file_output \
        --evaluation_method $evaluation_method \

After the above processing, your data directory will look like this: 

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_gowalla
        |   ├── train.txt
        |   └── test.txt
        └── instance_gowalla
            ├── info.yaml
            ├── indices.pkl
            ├── indptr.pkl
            └── test.pkl

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

The following shell script runs a LightGCN model with ``XGCN.main.run_model`` module and 
reproduce the results on the gowalla dataset: 

.. code:: shell

    # set to your own paths: 
    all_data_root=/home/xxx/XGCN_data
    config_file_root=/home/xxx/XGCN_library/config

    dataset=gowalla
    model=LightGCN
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-full_graph-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method multi_pos_whole_graph \
        --file_val_set $data_root/test_set.pkl \
        --test_method multi_pos_whole_graph \
        --file_test_set $data_root/test_set.pkl \
        --str_num_total_samples num_users \
        --pos_sampler NodeBased_ObservedEdges_Sampler \
        --neg_sampler StrictNeg_Sampler \
        --epoch_sample_ratio 27.13 \
        --num_gcn_layers 4 \
        --L2_reg_weight 1e-4 --use_ego_emb_L2_reg 1 \
        --emb_lr 0.001 \
        --emb_dim 64 \
        --train_batch_size 2048 \
        --epochs 10 --val_freq 5 \
        --key_score_metric r20 --convergence_threshold 1000 \

The results will be around: Recall@20:0.1827, NDCG@20:0.1550
