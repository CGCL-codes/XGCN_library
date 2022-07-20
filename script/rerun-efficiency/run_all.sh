PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cuda:0'

bash pprgo_pokec.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
bash pprgo_xbox-3m.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
bash pprgo_livejournal.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE

bash xgcn_pokec.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
bash xgcn_xbox-3m.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
bash xgcn_livejournal.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE

bash lightgcn_pokec.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
bash lightgcn_xbox-3m.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
bash lightgcn_livejournal.sh $PROJECT_ROOT $ALL_DATA_ROOT $DEVICE
