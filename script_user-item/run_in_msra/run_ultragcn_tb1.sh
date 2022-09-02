source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='./xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'
DEVICE='cuda'
 
########################################
DATASET='taobao-1.6m'

lambda=0.2
gamma=1.5
reg=1e-5
num_neg=64
neg_weight=64
topk=8

bash xGCN/script_user-item/ultragcn/run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk

