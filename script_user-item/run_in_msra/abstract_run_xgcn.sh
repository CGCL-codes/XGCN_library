source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn

## arguments of this shell: 
## dataset reg gamma topk


PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'


DEVICE='cuda'

########################################
DATASET=$1
reg=$2
gamma=$3
topk=$4

bash xGCN/script_user-item/xgcn/run_xgcn_single.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $gamma $topk
