PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$3

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

#### prepare_ultragcn_data 
RESULTS_DIR='ultragcn/data'
python $PROJECT_ROOT'/'model/UltraGCN/prepare_ultragcn_data.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 100 \
