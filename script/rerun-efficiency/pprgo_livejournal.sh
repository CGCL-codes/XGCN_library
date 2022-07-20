PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='livejournal'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
PPR_DATA_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/ppr/undirected-top100'

################
SEED=1
RESULTS_DIR='pprgo/efficiency'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/pprgo-config.yaml' \
    --data_root $DATA_ROOT \
    --seed $SEED \
    --ppr_data_root $PPR_DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --device $DEVICE \
    --train_batch_size 1024 \
    --emb_lr 0.005 \
    --l2_reg_weight 0.0 \
    --loss_fn 'bpr_loss' \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 100 --epochs 100 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
