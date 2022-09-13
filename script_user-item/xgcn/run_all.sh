PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

DEVICE='cuda:1'

DATASET='taobao-1.6m'

reg=1e-5

gamma=0.1
topk=16

bash run_xgcn_single.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $gamma $topk
