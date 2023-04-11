Example: gowalla
======================

Here we present a user-item graph example. The used gowalla dataset can be found 
in our XGCN repository: ``data/raw_gowalla/``, which is copied from LightGCN's official code repository: 
https://github.com/gusye1234/LightGCN-PyTorch.


Prepare the Dataset Instance
-------------------------------

Let's first import some modules and functions:

.. code:: python

    import XGCN
    from XGCN.data import io, csr
    from XGCN.utils.utils import ensure_dir, set_random_seed
    import os.path as osp

We recommend to arrange the data with a clear directory structure. 
To get started, you may manually setup an ``XGCN_data`` (or other names you like) directory as follows: 
(It's recommended to put your ``XGCN_data`` somewhere else than in this repository.)

.. code:: 

    XGCN_data
    └── dataset
        └── raw_gowalla
            ├── train.txt
            └── test.txt

We'll use this directory to hold all the different datasets 
and models outputs. 
We refer to its path as ``all_data_root`` in our python code and shell scripts: 

.. code:: python

    >>> # set your own all_data_root:
    >>> all_data_root = '/home/xxx/XGCN_data'

Load the graph from ``train.txt`` and convert it to CSR format:

.. code:: python

    >>> dataset = 'gowalla'
    >>> raw_data_root = osp.join(all_data_root, 'dataset/raw_' + dataset)
    >>> E_src, E_dst = io.load_txt_adj_as_edges(osp.join(raw_data_root, 'train.txt'))
    >>> print(E_src)
    [    0     0     0 ... 29857 29857 29857]
    >>> print(E_dst)
    [   0    1    2 ... 1853  691  674]
    >>> info, indptr, indices = csr.from_edges_to_csr_with_info(E_src, E_dst, graph_type='user-item')
    # from_edges_to_csr ...
    # remove_repeated_edges ...
    ## 0 edges are removed
    >>> print(info)
    {'graph_type': 'user-item', 'num_users': 29858, 'num_items': 40981, 'num_nodes': 70839, 'num_edges': 810128}

It is a user-item graph, so we set the ``graph_type`` argument 
to ``'user-item'`` in function ``csr.from_edges_to_csr_with_info()``.

Load the provided test set:

.. code:: python

    test_set = io.from_txt_adj_to_adj_eval_set(osp.join(raw_data_root, 'test.txt'))

Save the complete dataset instance (the validation set is not provided, so we just use the test set for early-stop): 

.. code:: python

    >>> data_root = osp.join(all_data_root, 'dataset/instance_' + dataset)
    >>> ensure_dir(data_root)
    >>> io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(data_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(data_root, 'indices.pkl'), indices)
    >>> io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)


Run LightGCN
-----------------

The follow shell script run a LightGCN model with ``XGCN.main.run_model`` module and 
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
        --val_method MultiPosWholeGraph_Evaluator \
        --file_val_set $data_root/test_set.pkl \
        --test_method MultiPosWholeGraph_Evaluator \
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
