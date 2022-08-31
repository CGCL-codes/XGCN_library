PROJECT_ROOT='/home/songxiran/code/xGCN'
ALL_DATA_ROOT='/home/songxiran/data/social_and_user_item'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='amazon-book-1.5m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

#### prepare_ultragcn_data 
RESULTS_DIR='ultragcn/data'
python $PROJECT_ROOT'/'model/UltraGCN/prepare_ultragcn_data.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 100 \

DATASET='taobao-1.6m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

#### prepare_ultragcn_data 
RESULTS_DIR='ultragcn/data'
python $PROJECT_ROOT'/'model/UltraGCN/prepare_ultragcn_data.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 100 \
