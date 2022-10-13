PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$3

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

use_ssnet=$4
seed=$5

RESULTS_DIR="pprgo/[use_ssnet${use_ssnet}][seed${seed}]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT/main/eval.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $RESULTS_ROOT \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --file_output $RESULTS_ROOT'/test_.json' \
