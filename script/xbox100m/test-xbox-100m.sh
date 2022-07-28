PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-100m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

RESULTS_DIR='graphsage/[0]'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT/main/eval.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $RESULTS_ROOT \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges-5000.pkl' \
    --file_output $RESULTS_ROOT'/test.json' \


# # for debug on pokec:
# python $PROJECT_ROOT/main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'one_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/val_edges-1000.pkl' \
#     --file_output $RESULTS_ROOT'/val_results.json' \
