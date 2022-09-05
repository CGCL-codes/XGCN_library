source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn


## arguments of this shell: 
## dataset reg num_neg neg_weight

PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'
DEVICE='cuda'

DATASET=$1
reg=$2
num_neg=$3
neg_weight=$4

bash xGCN/script_user-item/simplex/run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight


