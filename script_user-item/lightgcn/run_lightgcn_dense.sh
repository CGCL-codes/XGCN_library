PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cuda:0'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='yelp2018'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
RESULTS_DIR='lightgcn/[0]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'lightgcn' \
    --config_file $CONFIG_ROOT'/lightgcn-config.yaml' --use_sparse 0 \
    --data_root $DATA_ROOT \
    --seed $SEED \
    --results_root $RESULTS_ROOT \
    --device $DEVICE \
    --validation_method 'multi_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/test-1000.pkl' \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --train_dl 'NodeBased_TrainDataLoader' \
    --key_score_metric 'r20' \
    --convergence_threshold 50 \
    --train_batch_size 2048 \
    --emb_lr 0.001 \
    --num_gcn_layers 3 \
    --stack_layers 1 \
    --l2_reg_weight 1e-4 \
    --loss_fn 'bpr_loss' \
