PROJECT_ROOT='/home/sxr/code/www23/xGCN'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='amazon-book'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET/'item2item_graph'

python $PROJECT_ROOT/data/filter_edges/filter_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $ALL_DATASETS_ROOT'/instance_'$DATASET/'instance_item2item_graph' \
    --dataset_name $DATASET \
