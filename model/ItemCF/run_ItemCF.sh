PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'
# ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

DATASET='yelp2018'
DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

RESULTS_DIR='ItemCF/[1]'
RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$RESULTS_DIR


python run_ItemCF.py $PROJECT_ROOT \
    --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --test_method 'multi_pos_whole_graph' \
    --file_test $DATA_ROOT'/test.pkl' \
    --use_degree_norm 1 \


# DATASET='LFM'
# DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

# RESULTS_DIR='ItemCF/[0]'
# RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$RESULTS_DIR


# python run_ItemCF.py $PROJECT_ROOT \
#     --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --file_test $DATA_ROOT'/val.pkl' \
#     # --file_test $DATA_ROOT'/val_edges-1000.pkl' \

# DATASET='taobao-1.6m'
# DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

# RESULTS_DIR='ItemCF/[1]'
# RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$RESULTS_DIR


# python run_ItemCF.py $PROJECT_ROOT \
#     --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
#     --test_method 'one_pos_whole_graph' \
#     --file_test $DATA_ROOT'/test_edges.pkl' \
#     --use_degree_norm 1 \
#     # --file_test $DATA_ROOT'/val_edges-1000.pkl' \
