PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='livejournal'
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p10.pkl' \
    --file_output $DATA_ROOT'/test-p10-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p20.pkl' \
    --file_output $DATA_ROOT'/test-p20-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p30.pkl' \
    --file_output $DATA_ROOT'/test-p30-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p40.pkl' \
    --file_output $DATA_ROOT'/test-p40-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p50.pkl' \
    --file_output $DATA_ROOT'/test-p50-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p60.pkl' \
    --file_output $DATA_ROOT'/test-p60-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p70.pkl' \
    --file_output $DATA_ROOT'/test-p70-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p80.pkl' \
    --file_output $DATA_ROOT'/test-p80-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p90.pkl' \
    --file_output $DATA_ROOT'/test-p90-1-999.pkl' \
    --num_neg 999 \

python $PROJECT_ROOT/data/add_sampled_neg_for_pos_edges.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input  $DATA_ROOT'/test-p100.pkl' \
    --file_output $DATA_ROOT'/test-p100-1-999.pkl' \
    --num_neg 999 \
