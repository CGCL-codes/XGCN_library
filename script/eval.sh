PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'


CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='pokec'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

RESULTS_DIR='pprgo/[new_ppr][lr0.005][bpr][reg0]'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python ../main/eval.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $RESULTS_ROOT \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --file_output $RESULTS_ROOT'/test.json' \

# PROJECT_ROOT='/home/sxr/code/xgcn'
# ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'


# CONFIG_ROOT=$PROJECT_ROOT'/config'
# ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
# ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

# DATASET=$1

# DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

# # RESULTS_DIR=$2
# # RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR
# RESULTS_ROOT=$2

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p10.pkl' \
#     --file_output $RESULTS_ROOT'/test-p10.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p20.pkl' \
#     --file_output $RESULTS_ROOT'/test-p20.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p30.pkl' \
#     --file_output $RESULTS_ROOT'/test-p30.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p40.pkl' \
#     --file_output $RESULTS_ROOT'/test-p40.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p50.pkl' \
#     --file_output $RESULTS_ROOT'/test-p50.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p60.pkl' \
#     --file_output $RESULTS_ROOT'/test-p60.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p70.pkl' \
#     --file_output $RESULTS_ROOT'/test-p70.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p80.pkl' \
#     --file_output $RESULTS_ROOT'/test-p80.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p90.pkl' \
#     --file_output $RESULTS_ROOT'/test-p90.json' \

# python ../main/eval.py $PROJECT_ROOT \
#     --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
#     --results_root $RESULTS_ROOT \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test-p100.pkl' \
#     --file_output $RESULTS_ROOT'/test-p100.json' \
