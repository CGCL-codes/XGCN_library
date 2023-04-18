.. _user_guide-training_and_evaluation-model_training:

Model Training
======================

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