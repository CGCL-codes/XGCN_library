source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

# PROJECT_ROOT='/media/xreco/jianxun/xGCN'
# ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

########################################
DATASET=$1
prop_times=$2

bash xGCN/script/run_in_msra/run_propagation.sh $PROJECT_ROOT $ALL_DATA_ROOT \
    $DATASET $prop_times \
