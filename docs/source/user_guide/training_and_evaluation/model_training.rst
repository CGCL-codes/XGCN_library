.. _user_guide-training_and_evaluation-model_training:

Model Training
======================

Train from scratch
------------------------------

There are three steps to train a model: 

(1) Prepare the ``config`` Dict, which contains all the needed arguments. 

(2) Create the model: ``model = XGCN.create_model(config)``. The ``results_root`` directory will be automatically created if it does not exist. 

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
For example, ``script/examples/facebook/run_xGCN.sh``: 

.. code:: bash

    # set to your own path:
    all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
    config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

    dataset=facebook
    model=xGCN
    seed=0
    device='cuda:0'
    emb_table_device=$device
    forward_device=$device
    out_emb_table_device=$device

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    # file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/model/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_method one_pos_k_neg \
        --file_val_set $data_root/val-one_pos_k_neg.pkl \
        --key_score_metric r20 \
        --test_method multi_pos_whole_graph \
        --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
        --emb_table_device $emb_table_device \
        --forward_device $forward_device \
        --out_emb_table_device $out_emb_table_device \
        # --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \

To run a script, you only need to modify ``all_data_root`` and 
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
In this case, please specify the previously saved ``config.yaml`` and call the 
``XGCN.load_model()`` function: 

.. code:: python

    config = io.load_yaml(...)  # the previously saved config.yaml
    config['emb_lr'] = 0.0001   # change some hyper-paramenters

    model = XGCN.load_model(config)  # load the saved model      
    model.fit()                      # training on the new hyper-paramenters
    new_resutls = model.test()