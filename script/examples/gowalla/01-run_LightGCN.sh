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
    