PROJECT_ROOT='/home/sxr/code/www23/xGCN'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='gowalla'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python $PROJECT_ROOT/data/generate_item2item_graph/generate_item2item_graph.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $DATA_ROOT/item2item_graph_with_degree_norm \
    --use_degree_norm 1 \

DATASET='yelp2018'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python $PROJECT_ROOT/data/generate_item2item_graph/generate_item2item_graph.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $DATA_ROOT/item2item_graph_with_degree_norm \
    --use_degree_norm 1 \
