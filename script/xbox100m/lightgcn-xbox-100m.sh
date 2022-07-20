PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cpu'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-100m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
RESULTS_DIR='lightgcn/[0]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'block_lightgcn' --train_dl 'EdgeBased_Sampling_Block_TrainDataLoader' \
    --config_file $CONFIG_ROOT'/lightgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_k_neg' \
    --mask_nei_when_validation 0 \
    --file_validation $DATA_ROOT'/test-1-99.pkl' --key_score_metric 'n20'  \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges-5000.pkl' \
    --train_batch_size 1024 \
    --emb_dim 32 \
    --epochs 120 --convergence_threshold 2 \
    --edge_sample_ratio 0.01 \
    --num_gcn_layers 1 --num_layer_sample '[32,]' --num_workers 4 \
    --l2_reg_weight 0 \
    --loss_fn 'bpr_loss' --num_neg 1 \
    --use_sparse 1 \

    # --validation_method 'one_pos_whole_graph' \
    # --mask_nei_when_validation 1 \
    # --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    # --test_method 'one_pos_whole_graph' \
    # --mask_nei_when_test 1 \
    # --file_test $DATA_ROOT'/val_edges-1000.pkl' \
