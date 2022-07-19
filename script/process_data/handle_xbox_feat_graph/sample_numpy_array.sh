PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='xbox-feat-1m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

# python $PROJECT_ROOT'/'data/sample_numpy_array.py $PROJECT_ROOT \
#     --file_input $DATA_ROOT'/valid_edges.pkl' \
#     --file_output $DATA_ROOT'/valid_edges-1000.pkl' \
#     --num_sample 1000 \

# python $PROJECT_ROOT'/'data/sample_numpy_array.py $PROJECT_ROOT \
#     --file_input $DATA_ROOT'/test_edges.pkl' \
#     --file_output $DATA_ROOT'/test_edges-1000.pkl' \
#     --num_sample 1000 \


DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET/'user_game_graph'

python $PROJECT_ROOT'/'data/sample_numpy_array.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/valid_edges.pkl' \
    --file_output $DATA_ROOT'/valid_edges-1000.pkl' \
    --num_sample 1000 \

python $PROJECT_ROOT'/'data/sample_numpy_array.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/test_edges.pkl' \
    --file_output $DATA_ROOT'/test_edges-1000.pkl' \
    --num_sample 1000 \
