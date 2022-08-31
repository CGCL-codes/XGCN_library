DEVICE='cuda:2'

DATASET='amazon-book-1.5m'

reg=1e-2
num_neg=64
neg_weight=64

bash run_simplex.sh $DEVICE $DATASET $reg $num_neg $neg_weight


DATASET='taobao-1.6m'

reg=1e-3
num_neg=64
neg_weight=64

bash run_simplex.sh $DEVICE $DATASET $reg $num_neg $neg_weight
