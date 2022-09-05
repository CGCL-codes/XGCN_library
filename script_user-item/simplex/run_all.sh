PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cuda'

###################################
DATASET='amazon-book-1.5m'

reg=1e-3
num_neg=64
neg_weight=64

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='amazon-book-1.5m'

reg=1e-3
num_neg=256
neg_weight=64

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='amazon-book-1.5m'

reg=1e-3
num_neg=256
neg_weight=256

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='amazon-book-1.5m'

reg=1e-3
num_neg=512
neg_weight=128

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='amazon-book-1.5m'

reg=1e-3
num_neg=512
neg_weight=512

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='taobao-1.6m'

reg=1e-4
num_neg=64
neg_weight=64

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='taobao-1.6m'

reg=1e-4
num_neg=256
neg_weight=64

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='taobao-1.6m'

reg=1e-4
num_neg=256
neg_weight=256

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='taobao-1.6m'

reg=1e-4
num_neg=512
neg_weight=128

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight

###################################
DATASET='taobao-1.6m'

reg=1e-4
num_neg=512
neg_weight=512

bash run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight
