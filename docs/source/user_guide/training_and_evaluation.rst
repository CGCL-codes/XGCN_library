Training and Evaluation
============================

Once the dataset instance is generated (for dataset instance generation, please refer to the previous section), 
you can run models with XGCN's APIs: 

.. code:: python

    model = XGCN.create_model(config)
    model.fit()

In this section, we are going to introduce:

* **How to set model configurations**
* **How to train a model**
* **How to evaluate a model**
* **Model inference APIs**

---------------------------
Model Configuration
---------------------------

The model configuration in XGCN is basically a Python Dict containing all the setting parameters. 
XGCN supports parsing model configurations from command line arguments and ``.yaml`` files. 
You can also manually write a Dict with all the parameters in a python script. 


Configuration Template
---------------------------

Directory ``config/`` includes ``.yaml`` configuration file templates for all the models. 
Each file contains **all** the arguments needed to run a model. 
A typical ``.yaml`` configuration file is like this:

.. code:: yaml

    # Dataset/Results root
    data_root: ""       # root of the dataset instance
    results_root: ""    # root for model outputs, training record, and evaluation results

    # Trainer configuration
    epochs: 200
    use_validation_for_early_stop: 1
    val_freq: 1
    key_score_metric: r100
    convergence_threshold: 20
    val_method: ""
    val_batch_size: 256
    file_val_set: ""

    # Testing configuration  (required for model.test())
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

    # DataLoader configuration
    Dataset_type: BlockDataset
    num_workers: 0
    num_gcn_layers: 2
    train_num_layer_sample: "[10, 20]"
    NodeListDataset_type: LinkDataset
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    epoch_sample_ratio: 0.1

    # Model configuration
    model: GraphSAGE
    seed: 1999
    graph_device: "cuda:0"
    emb_table_device: "cuda:0"
    gnn_device: "cuda:0"
    out_emb_table_device: "cuda:0"
    forward_mode: sample
    infer_num_layer_sample: "[10, 20]"
    from_pretrained: 0
    file_pretrained_emb: ""
    freeze_emb: 0
    use_sparse: 0
    emb_dim: 64 
    emb_init_std: 0.1
    emb_lr: 0.005
    gnn_arch: "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]"
    gnn_lr: 0.01
    loss_type: bpr
    L2_reg_weight: 0.0

The configuration consists of five parts:

(1) **Dataset/Results root**. 
Specifies the dataset instance root and the directory to save the outputs during the model training. Note that when calling the ``XGCN.create_model(config)`` function, the 'results_root' directory will be automatically created if it does not exist. 

(2) **Trainer configuration**. 
Specifies the configuration about training loop control, e.g. ``epochs``. 

(3) **Testing configuration**. 
Specifies the configurations about model testing. This field is required for ``model.test()`` function. 

(4) **DataLoader configuration**. 
Specifies the dataloader for training. 

(5) **Model configuration**. 
Specifies the model configuration such as hyper-parameters. 


Load config from yaml file
---------------------------

We can load a ``.yaml`` configuration file with ``XGCN.data.io`` module:

.. code:: python

    import XGCN
    from XGCN.data import io

    config = io.load_yaml('config.yaml')  # load template
    config['data_root'] = ...             # add/modify some configurations


Parse config from command line
--------------------------------

We also provide a ``parse_arguments()`` to parse command line arguments: 

.. code:: python

    import XGCN
    from XGCN.utils.parse_arguments import parse_arguments

    config = parse_arguments()


You can specify a ``.yaml`` configuration file with ``--config_file``. 
Note that a configuration file is not a necessity for the ``parse_arguments()`` function 
and has lower priority when the same command line argument is given. 


------------------
Model Training
------------------

There are three steps to train a model: 

(1) Prepare the ``config`` Dict, which contains all the needed arguments. 

(2) Create the model: ``model = XGCN.create_model(config)``. The 'results_root' directory will be automatically created if it does not exist. 

(3) Start training: ``model.fit()``. The best model on the validation set and the training information will be save at ``results_root``. 

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

We provide shell scripts to run all the models in ``script/examples``.
For example, ``run_xGCN-facebook.sh``: 

.. code:: bash

    # modify to your own paths:
    all_data_root=/home/xxx/XGCN_data
    config_file_root=/home/xxx/XGCN_library/config  # path to the config file templates

    dataset=facebook
    model=xGCN
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method one_pos_k_neg --val_batch_size 256 \
        --file_val_set $data_root/val-one_pos_k_neg.pkl \
        --test_method multi_pos_whole_graph --test_batch_size 256 \
        --file_test_set $data_root/test-multi_pos_whole_graph.pkl \

To run a shell script, you only need to modify ``all_data_root`` and 
``config_file_root`` to your own paths. 

Once a model is trained, the output data will be saved at ``results_root``: 

.. code:: 

    XGCN_data
    └── model_output
        └── facebook
            └── xGCN
                └── [seed0]
                    ├── model (directory)       # the best model on the validation set
                    ├── config.yaml             # configurations of the running
                    ├── mean_time.json          # time consumption information in seconds
                    ├── test_results.json       # test results
                    ├── train_record_best.json  # validation results of the best epoch
                    └── train_record.txt        # validation results of all the epochs


Load and continue to train
------------------------------

XGCN can also load trained models and continue to train. 
In this case please specify the previously saved ``config.yaml`` and call the 
``XGCN.load_model()`` function: 

.. code:: python

    config = io.load_yaml(...)  # the previously saved config.yaml
    config['emb_lr'] = 0.0001   # change some hyper-paramenters

    model = XGCN.load_model(config)  # load the saved model      
    model.fit()                      # training on the new hyper-paramenters
    new_resutls = model.test()


--------------------
Model Evaluation
--------------------

To test a model, you can just call ``model.test()``, 
it executes the default testing setting configurations in ``config``: 

.. code:: python

    config = {
        'data_root': ..., 'results_root': ..., 
        'model': 'xGCN', 'seed': 1999, 
        ...,
        'test_method': 'OnePosKNeg_Evaluator', 
        'test_batch_size': 256, 'file_test_set': ...,
        ...
    }
    model = XGCN.create_model(config)
    model.fit()  
    results = model.test()

Or you can specify other test sets:

.. code:: python

    test_config = {
        'test_method': 'multi_pos_whole_graph',
        'test_batch_size': 256,
        'file_test_set': ... 
    }
    results = model.test(test_config)

The function receives a Dict containing three arguments: 

* ``test_method``: specifices the evaluation method. Available options: 'one_pos_k_neg', 'one_pos_whole_graph', and 'multi_pos_whole_graph'. 

* ``test_batch_size``: specifices the batch size. 

* ``file_test_set``: specifices the file of the processed evaluation set. 


------------------
Model Inference
------------------

XGCN provides some model inference APIs: 

.. code:: python

    # infer scores given a source node and one or more target nodes:
    target_score = model.infer_target_score(
        src=5, 
        target=torch.LongTensor(101, 102, 103)
    )

    # infer top-k recommendations for a source node
    score, topk_node = model.infer_topk(k=100, src=5, mask_nei=True)

    # save the output embeddings as a text file
    model.save_emb_as_txt(filename='out_emb_table.txt')
