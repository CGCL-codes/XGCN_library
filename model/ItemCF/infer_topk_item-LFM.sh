
PROJECT_ROOT="/media/xreco/jianxun/xGCN"
ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

if [ $# -gt 1 ]
  then 
    PROJECT_ROOT=$1
    ALL_DATA_ROOT=$2 
fi


DATASET='LFM'
DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

RESULTS_DIR='ItemCF/[0]'
RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$RESULTS_DIR

topk=100
use_degree_norm=1
FILE_INPUT=$DATA_ROOT"/test_user_ids.txt"  # each line contains a user id
FILE_OUTPUT=$RESULTS_ROOT"/top${topk}-item-reco.txt"  # each line contains topk item id (id starts from 0), seperated by blank

cd $PROJECT_ROOT
python model/ItemCF/infer_topk_item.py $PROJECT_ROOT \
    --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --topk=$topk \
    --file_input=$FILE_INPUT \
    --file_output=$FILE_OUTPUT \
    --use_degree_norm=$use_degree_norm 
