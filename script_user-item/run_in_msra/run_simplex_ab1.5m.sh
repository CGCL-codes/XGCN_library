source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn


PROJECT_ROOT='xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

DEVICE='cuda'

DATASET='amazon-book-1.5m'

reg=1e-2
num_neg=64
neg_weight=64

bash xGCN/script_user-item/simplex/run_simplex.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE $DATASET $reg $num_neg $neg_weight


