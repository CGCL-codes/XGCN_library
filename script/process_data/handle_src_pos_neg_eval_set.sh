PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET=$3

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
FILE_INPUT=$DATA_ROOT'/val_edges-1000.txt'
FILE_OUTPUT=$DATA_ROOT'/val_edges-1000.pkl'

python $PROJECT_ROOT'/'data/handle_src_pos_neg_eval_set.py $PROJECT_ROOT \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \
