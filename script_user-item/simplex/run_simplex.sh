PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$4

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1

reg=$5
num_neg=$6
neg_weight=$7

RESULTS_DIR="simplex/[reg$reg][neg$num_neg-weight$neg_weight]"

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/simplex-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --train_batch_size 4096 \
    --emb_lr 0.001 \
    --l2_reg_weight $reg \
    --num_neg $num_neg \
    --neg_weight $neg_weight \
    --margin 0.4 \
    --theta 0.5 \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 100 --epochs 1000 \
    --use_sparse 0 \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
