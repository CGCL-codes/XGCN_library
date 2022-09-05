source /opt/conda/bin/activate
conda env create --file=env/requirements.xgcn.yaml
conda activate xgcn

PROJECT_ROOT='./xGCN'
ALL_DATA_ROOT='/home/jialia/ds/social_and_user_item'

bash xGCN/script_user-item/ultragcn/prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT 'amazon-book-1.5m'
bash xGCN/script_user-item/ultragcn/prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT 'taobao-1.6m'
