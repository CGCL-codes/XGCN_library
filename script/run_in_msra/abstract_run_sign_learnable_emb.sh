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

num_gcn_layers=$3
num_layer_sample=$4
num_dnn_layers=$5

# bash $PROJECT_ROOT/script/run_in_msra/run_sign_learnable_emb.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $seed \
#     $num_gcn_layers $num_layer_sample $num_dnn_layers

bash xGCN/script/run_in_msra/run_sign_learnable_emb.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $seed \
    $num_gcn_layers $num_dnn_layers
