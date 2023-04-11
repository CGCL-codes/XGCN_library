Example: facebook
======================

Let's begin with a small social network dataset: facebook.
The data is included in the XGCN repository: ``data/raw_facebook``. 
You can also download it from SNAP: http://snap.stanford.edu/data/facebook_combined.txt.gz .


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
        └── raw_facebook
            └── facebook_combined.txt

We'll use this directory to hold all the different datasets 
and models outputs. 
We refer to its path as ``all_data_root`` in our python code and shell scripts: 

.. code:: python

    >>> # set your own all_data_root:
    >>> all_data_root = '/home/xxx/XGCN_data'

To get started, let's load the raw graph from the text file and convert it 
to CSR format. The graph is a social network, so we set the ``graph_type`` argument 
to ``'homo'`` in function ``csr.from_edges_to_csr_with_info()``. 

.. code:: python

    >>> dataset = 'facebook'
    >>> raw_data_root = osp.join(all_data_root, 'dataset/raw_' + dataset)
    >>> file_raw_graph = osp.join(raw_data_root, 'facebook_combined.txt')
    >>> E_src, E_dst = io.load_txt_edges(file_raw_graph)
    >>> print(E_src)
    [   0    0    0 ... 4027 4027 4031]
    >>> print(E_dst)
    [   1    2    3 ... 4032 4038 4038]
    >>> info, indptr, indices = csr.from_edges_to_csr_with_info(E_src, E_dst, graph_type='homo')
    # from_edges_to_csr ...
    # remove_repeated_edges ...
    ## 0 edges are removed
    >>> print(info)
    {'graph_type': 'homo', 'num_nodes': 4039, 'num_edges': 88234}

Loading large graphs from text files can be time-consuming 
(though the facebook graph here is a small one), 
we can cache the graph using ``io.save_pickle``: 

.. code:: python

    >>> raw_csr_root = osp.join(raw_data_root, 'csr')
    >>> ensure_dir(raw_csr_root)  # mkdir if not exists
    >>> io.save_yaml(osp.join(raw_csr_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(raw_csr_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(raw_csr_root, 'indices.pkl'), indices)

Assume that we don't have existing evaluation set 
and want to split some edges for model evaluation: 

.. code:: python

    >>> set_random_seed(1999)
    >>> num_sample = 10_000       # number of edges to split
    >>> min_src_out_degree = 3    # guarantee the minimum out-degree of a source node
    >>> min_dst_in_degree = 1     # guarantee the minimum in-degree of a destination node
    >>> indptr, indices, pos_edges = XGCN.data.split.split_edges(indptr, indices, num_sample, min_src_out_degree, min_dst_in_degree)
    sampling edges 9999/10000 (99.99%)
    num sampled edges: 10000
    >>> info['num_edges'] = len(indices)  # number of edges after the split
    >>> print(info)
    {'graph_type': 'homo', 'num_nodes': 4039, 'num_edges': 78234}

We get all the positive edges: ``pos_edges``, let's divide them for 
validation set and test set, and we'll use the "whole-graph-multi-pos" evaluation method:

.. code:: python

    >>> num_validation = 2000
    >>> val_edges = pos_edges[:num_validation]
    >>> test_edges = pos_edges[num_validation:]
    >>> val_set = XGCN.data.split.from_edges_to_adj_eval_set(val_edges)
    >>> test_set = XGCN.data.split.from_edges_to_adj_eval_set(test_edges)

Now we have already generated a complete dataset instance, let's save it:

.. code:: python

    >>> data_root = osp.join(all_data_root, 'dataset/instance_' + dataset)
    >>> ensure_dir(data_root)
    >>> io.save_yaml(osp.join(data_root, 'info.yaml'), info)
    >>> io.save_pickle(osp.join(data_root, 'indptr.pkl'), indptr)
    >>> io.save_pickle(osp.join(data_root, 'indices.pkl'), indices)
    >>> io.save_pickle(osp.join(data_root, 'pos_edges.pkl'), pos_edges)
    >>> io.save_pickle(osp.join(data_root, 'val_set.pkl'), val_set)
    >>> io.save_pickle(osp.join(data_root, 'test_set.pkl'), test_set)

Here we also save the ``pos_edges``, so you can use it to make evaluation sets for 
"one-pos-k-neg" or "whole-graph-one-pos" method by concatenating some randomly 
sampled negative nodes. 

If you have done the above steps successfully, your data directory will look like follows: 

.. code:: 

    XGCN_data
    └── dataset
        ├── raw_facebook
        |   ├── facebook_combined.txt
        |   └── csr
        |       ├── indices.pkl
        |       ├── indptr.pkl
        |       └── info.yaml
        └── instance_facebook
            ├── indices.pkl
            ├── indptr.pkl
            ├── info.yaml
            ├── pos_edges.pkl
            ├── test_set.pkl
            └── val_set.pkl


Run xGCN from Command Line Interface
-------------------------------------

We can run a model by using XGCN's command line interface. 
XGCN supports parsing model configurations from command line arguments and ``.yaml`` files.
Directory ``config/`` includes ``.yaml`` configuration file templates for all the models.

Take the xGCN model as an example, you can write a shell script
named ``run_xGCN-facebook.sh`` like this: 

.. code:: shell

    # set to your own paths: 
    all_data_root=/home/xxx/XGCN_data
    config_file_root=/home/xxx/XGCN_library/config

    dataset=facebook
    model=xGCN
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method MultiPosWholeGraph_Evaluator \
        --file_val_set $data_root/val_set.pkl \
        --test_method MultiPosWholeGraph_Evaluator \
        --file_test_set $data_root/test_set.pkl \

If you want to use a ``.yaml`` configuration file, specify the path 
with the command line argument ``--config_file``. 
Note that a ``.yaml`` file is not a necessity of running the code and has lower 
priority when the same command line argument is given. 

Run the script: ``bash run_xGCN-facebook.sh``, and when the training is converged (i.e. the 
validation score has not increased for a specified number of epochs), the best model on the 
validation set will be loaded and model testing will begin. 

After the whole training & testing process, your ``results_root`` directory will be like follows: 

.. code:: 

    XGCN_data
    └── model_output
        └── facebook
            └── xGCN
                └── [seed0]
                    ├── config.yaml             # configurations of the running
                    ├── mean_time.json          # time consumption information in seconds
                    ├── out_emb_table.pt        # the best output embeddings on validation set
                    ├── test_results.json       # test results
                    ├── train_record_best.json  # validation results of the best epoch
                    └── train_record.txt        # validation results and losses during training


Run xGCN from Python Script
-------------------------------------

We can also easily run a model in your own Python scripts 
by using XGCN's API functions. 

For example, create a Python script named ``run.py`` with the following contents: 

.. code:: python

    import XGCN
    from XGCN.utils.parse_arguments import parse_arguments

    def main():
        
        config = parse_arguments()  # the config is just a python Dict

        model = XGCN.create_model(config)
        
        model.fit()  # train & test

    if __name__ == '__main__':
        
        main()

To run it, for convenience, we also create a shell script ``run.sh`` 
(almost the same as the previous ``run_xGCN-facebook.sh``): 

.. code:: shell

    # set to your own paths: 
    all_data_root=/home/xxx/XGCN_data
    config_file_root=/home/xxx/XGCN_library/config

    dataset=facebook
    model=xGCN
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    python run.py --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method MultiPosWholeGraph_Evaluator \
        --file_val_set $data_root/val_set.pkl \
        --test_method MultiPosWholeGraph_Evaluator \
        --file_test_set $data_root/test_set.pkl \

Then you can run your Python code with ``bash run.sh``. 

XGCN's ``BaseEmbeddingModel`` class provides some useful functions for model inference, 
you can call these functions when the training is done, for example: 

.. code:: python

    ...
    model.fit()

    # infer scores given a source node and one or more target nodes:
    target_score = model.infer_target_score(
        src=5, 
        target=torch.LongTensor(101, 102, 103)
    )

    # infer top-k recommendations for a source node
    score, topk_node = model.infer_topk(k=100, src=5, mask_nei=True)

    # save the output embeddings as a text file
    model.save_emb_as_txt(filename='out_emb_table.txt')
