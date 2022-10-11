source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

# PROJECT_ROOT='/media/xreco/jianxun/xGCN'
# ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'
DEVICE='cuda'

########################################
DATASET=$1
seed=$2
num_gcn_layers=$3
reg=$4
use_dnn=$5

bash xGCN/script/run_in_msra/run_lightgcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE \
    $DATASET $seed \
    $num_gcn_layers $reg $use_dnn \
