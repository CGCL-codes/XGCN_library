PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET=$3

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
FILE_INPUT=$DATA_ROOT'/test.txt'
FILE_OUTPUT=$DATA_ROOT'/test.pkl'

python $PROJECT_ROOT'/'data/handle_adj_eval_set.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'social' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \
