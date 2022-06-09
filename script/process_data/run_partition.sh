PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='pokec'
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

NUM_PART=8
PART_METHOD='metis'
# PART_METHOD='random'
RESULTS_DIR='partition/'$PART_METHOD$NUM_PART
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/run_partition.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --part_method $PART_METHOD \
    --num_part $NUM_PART \
