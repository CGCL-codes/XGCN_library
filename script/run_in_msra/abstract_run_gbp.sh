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
walk_length=$3
alpha=$4
rmax_ratio=$5

bash $PROJECT_ROOT/script/run_in_msra/run_gbp.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $seed \
    $walk_length $alpha $rmax_ratio
