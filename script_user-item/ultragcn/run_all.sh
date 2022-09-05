PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

bash prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT 'amazon-book-1.5m'
bash prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT 'taobao-1.6m'

DEVICE='cuda'

########################################
DATASET='amazon-book-1.5m'

lambda=0.8
gamma=3.5
reg=1e-5
num_neg=256
neg_weight=256
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk

########################################
DATASET='amazon-book-1.5m'

lambda=0.8
gamma=3.5
reg=1e-5
num_neg=512
neg_weight=128
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk

########################################
DATASET='taobao-1.6m'

lambda=0.8
gamma=10.0
reg=1e-5
num_neg=64
neg_weight=64
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk

########################################
DATASET='taobao-1.6m'

lambda=0.8
gamma=50.0
reg=1e-5
num_neg=64
neg_weight=64
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk

########################################
DATASET='taobao-1.6m'

lambda=0.8
gamma=100.0
reg=1e-5
num_neg=512
neg_weight=512
topk=8

bash run_ultragcn.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk
