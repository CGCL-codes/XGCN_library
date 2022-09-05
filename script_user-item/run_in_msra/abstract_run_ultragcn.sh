source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

## arguments of this shell: 
## dataset lambda gamma reg num_neg neg_weight topk


PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'


DEVICE='cuda'

########################################
DATASET=$1
lambda=$2
gamma=$3
reg=$4
num_neg=$5
neg_weight=$6
topk=$7

bash xGCN/script_user-item/ultragcn/run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk
