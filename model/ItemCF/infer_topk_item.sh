PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'
# ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

DATASET='yelp2018'
DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

RESULTS_DIR='ItemCF/[0]'
RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$RESULTS_DIR

topk=10
FILE_INPUT=$RESULTS_ROOT"/uid.txt"  # each line contains a user id
FILE_OUTPUT=$RESULTS_ROOT"/top${topk}-item.txt"  # each line contains topk item id (id starts from 0), seperated by blank

python infer_topk_item.py $PROJECT_ROOT \
    --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --topk 10 \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \
    --use_degree_norm=1
