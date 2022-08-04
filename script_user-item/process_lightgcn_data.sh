PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

######################################
DATASET='gowalla'

LIGHTGCN_DATA_ROOT='/media/xreco/DEV/xiran/code/LightGCN-PyTorch/data/'$DATASET
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

FILE_INPUT=$LIGHTGCN_DATA_ROOT'/train.txt'

python $PROJECT_ROOT'/'data/handle_adj_graph_txt.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \


FILE_INPUT=$LIGHTGCN_DATA_ROOT'/test.txt'
FILE_OUTPUT=$DATA_ROOT'/test.pkl'

python $PROJECT_ROOT'/'data/handle_adj_eval_set.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \


python $PROJECT_ROOT'/'data/sample_multi_pos_eval_set.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/test.pkl' \
    --file_output $DATA_ROOT'/test-1000.pkl' \
    --num_sample 1000 \


######################################
DATASET='yelp2018'

LIGHTGCN_DATA_ROOT='/media/xreco/DEV/xiran/code/LightGCN-PyTorch/data/'$DATASET
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

FILE_INPUT=$LIGHTGCN_DATA_ROOT'/train.txt'

python $PROJECT_ROOT'/'data/handle_adj_graph_txt.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \


FILE_INPUT=$LIGHTGCN_DATA_ROOT'/test.txt'
FILE_OUTPUT=$DATA_ROOT'/test.pkl'

python $PROJECT_ROOT'/'data/handle_adj_eval_set.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \


python $PROJECT_ROOT'/'data/sample_multi_pos_eval_set.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/test.pkl' \
    --file_output $DATA_ROOT'/test-1000.pkl' \
    --num_sample 1000 \


######################################
DATASET='amazon-book'

LIGHTGCN_DATA_ROOT='/media/xreco/DEV/xiran/code/LightGCN-PyTorch/data/'$DATASET
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

FILE_INPUT=$LIGHTGCN_DATA_ROOT'/train.txt'

python $PROJECT_ROOT'/'data/handle_adj_graph_txt.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \


FILE_INPUT=$LIGHTGCN_DATA_ROOT'/test.txt'
FILE_OUTPUT=$DATA_ROOT'/test.pkl'

python $PROJECT_ROOT'/'data/handle_adj_eval_set.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'user-item' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \


python $PROJECT_ROOT'/'data/sample_multi_pos_eval_set.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/test.pkl' \
    --file_output $DATA_ROOT'/test-1000.pkl' \
    --num_sample 1000 \
