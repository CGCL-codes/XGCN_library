PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-3m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
RESULTS_DIR='lightgcn/[best'$SEED']'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'lightgcn_dgl_block' --train_dl 'EdgeBased_Sampling_Block_TrainDataLoader' \
    --config_file $CONFIG_ROOT'/lightgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/validation.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 20 --epochs 200 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --num_gcn_layers 1 --num_layer_sample '[]' --num_workers 5 \
    --l2_reg_weight 0 \
    --stack_layers 0 \
    --loss_fn 'bpr_loss' --num_neg 1 \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
