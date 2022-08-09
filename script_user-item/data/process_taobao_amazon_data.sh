# PROJECT_ROOT='/media/xreco/jianxun/xGCN'
# ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'
PROJECT_ROOT='/home/jialia/reco/xGCN'
ALL_DATA_ROOT='/home/jialia/reco'


ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

######################################
DATASET='taobao-1.6m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
# FILE_INPUT='/media/xreco/DEV/xiran/data/user_item_lightgcn/datasets/instance_taobao-1.6m/train_edges.txt'
FILE_INPUT="$DATA_ROOT/train_edges.txt"

python $PROJECT_ROOT'/'data/handle_edge_list_txt.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \


FILE_INPUT="$DATA_ROOT/val_pos_edges.txt"
FILE_OUTPUT=$DATA_ROOT'/val_edges.pkl'

python $PROJECT_ROOT'/'data/handle_src_pos_neg_eval_set.py $PROJECT_ROOT \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \


python $PROJECT_ROOT'/'data/sample_numpy_array.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/val_edges.pkl' \
    --file_output $DATA_ROOT'/val_edges-1000.pkl' \
    --num_sample 1000 \



FILE_INPUT="$DATA_ROOT/test_pos_edges.txt"
FILE_OUTPUT=$DATA_ROOT'/test_edges.pkl'

python $PROJECT_ROOT'/'data/handle_src_pos_neg_eval_set.py $PROJECT_ROOT \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \
