PROJECT_ROOT='/home/jialia/reco/xGCN'
ALL_DATA_ROOT='/home/jialia/reco'

DEVICE='cuda:0'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='taobao-1.6m'  ## 'taobao-1.6m' 'amazon-book-1.5m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
RESULTS_DIR='lightgcn/[sparse]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'block_lightgcn' --train_dl 'EdgeBased_Sampling_Block_TrainDataLoader' \
    --config_file $CONFIG_ROOT'/lightgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'multi_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val-1000.pkl' \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 20 \
    --train_batch_size 2048 \
    --num_gcn_layers 2 --num_layer_sample '[]' --num_workers 4 \
    --l2_reg_weight 1e-3 \
    --loss_fn 'bpr_loss' --num_neg 1 \

# 2 layer, each layer sample 16: --num_layer_sample '[16, 16]'
