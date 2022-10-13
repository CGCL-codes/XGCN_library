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
T=$2
K=$3
use_special_dnn=$4
dnn_arch=$5

bash xGCN/script/run_in_msra/run_xgcn.sh $PROJECT_ROOT $ALL_DATA_ROOT \
    $DEVICE $DATASET \
    $T $K $use_special_dnn $dnn_arch \
