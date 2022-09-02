PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

bash prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT 'amazon-book-1.5m'
bash prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT 'taobao-1.6m'

DEVICE='cuda'

########################################
DATASET='amazon-book-1.5m'

lambda=0.8
gamma=3.5
reg=1e-5
num_neg=256
neg_weight=64
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk


########################################
DATASET='amazon-book-1.5m'

lambda=0.8
gamma=3.5
reg=1e-5
num_neg=256
neg_weight=16
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk


########################################
DATASET='taobao-1.6m'

lambda=0.2
gamma=1.5
reg=1e-5
num_neg=64
neg_weight=64
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk


########################################
DATASET='taobao-1.6m'

lambda=1.4
gamma=1.5
reg=1e-5
num_neg=64
neg_weight=64
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk
