source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn 

PROJECT_ROOT='./xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

DEVICE='cuda'



########################################
DATASET='amazon-book-1.5m'

lambda=0.8
gamma=3.5
reg=1e-5
num_neg=256
neg_weight=16
topk=8

bash xGCN/script_user-item/ultragcn/run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk
