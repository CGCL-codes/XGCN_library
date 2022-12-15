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

GAMLP_type=$3
num_gcn_layers=$4
hidden=$5
n_layers_1=$6
n_layers_2=${7}
pre_process=${8}
residual=${9}
bns=${10}

bash $PROJECT_ROOT/script/run_in_msra/run_gamlp.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $seed \
    $GAMLP_type $num_gcn_layers \
    $hidden $n_layers_1 $n_layers_2 $pre_process $residual $bns \
