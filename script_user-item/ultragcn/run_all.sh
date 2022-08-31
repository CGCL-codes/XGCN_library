bash prepare_ultragcn_data.sh

DEVICE='cuda:3'

DATASET='amazon-book-1.5m'

lambda=0.8
gamma=3.5
reg=1e-5
num_neg=256
neg_weight=256
topk=8

bash run_ultragcn.sh $DEVICE $DATASET $lambda $gamma $reg $num_neg $neg_weight $topk
