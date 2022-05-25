PROJECT_ROOT='/home/xxx/code/xgcn'
ALL_DATA_ROOT='/home/xxx/data/xgcn_data'

DEVICE='cuda:0'

DATASET='pokec'

MODEL='xgcn'

cd run_model

bash $MODEL'_'$DATASET'.sh' $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
