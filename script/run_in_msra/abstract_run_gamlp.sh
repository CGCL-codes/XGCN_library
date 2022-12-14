source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

# PROJECT_ROOT='/home/sxr/code/www23/xGCN'
# ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

DEVICE='cuda'

########################################
DATASET=$1
seed=$2

GAMLP_type=$3
num_gcn_layers=$4

# bash $PROJECT_ROOT/script/run_in_msra/run_gamlp.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $seed \
#     $GAMLP_type $num_gcn_layers

bash xGCN/script/run_in_msra/run_gamlp.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $seed \
    $GAMLP_type $num_gcn_layers
