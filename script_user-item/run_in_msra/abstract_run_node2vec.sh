source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn


## arguments of this shell: 
## dataset p q context_size

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'
DEVICE='cuda'

DATASET=$1
p=$2
q=$3
context_size=$4

bash xGCN/script/run_node2vec-social.sh $PROJECT_ROOT $ALL_DATA_ROOT \
    $DEVICE $DATASET \
    $p $q $context_size \
