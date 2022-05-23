PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

DEVICE='cuda:0'

DATASET='pokec'

MODEL='xgcn'

cd run_model

bash $MODEL'_'$DATASET'.sh' $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
