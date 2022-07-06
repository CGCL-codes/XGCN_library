PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='yelp2018'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python $PROJECT_ROOT'/'data/sample_multi_pos_eval_set.py $PROJECT_ROOT \
    --file_input $DATA_ROOT'/test.pkl' \
    --file_output $DATA_ROOT'/test-sampled.pkl' \
