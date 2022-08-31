DEVICE='cuda:2'

DATASET='amazon-book-1.5m'

reg=1e-4
num_neg=64
neg_weight=64

bash run_simplex.sh $DEVICE $DATASET $reg $num_neg $neg_weight


# reg=1e-4
# num_neg=64
# neg_weight=16

# bash run_simplex.sh $DEVICE $DATASET $reg $num_neg $neg_weight
