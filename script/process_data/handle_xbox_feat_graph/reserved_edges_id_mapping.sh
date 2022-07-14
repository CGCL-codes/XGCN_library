PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='xbox-feat-1m'

RAW_DATA_ROOT='/media/xreco/DEV/socgraph/usconsoleall_sampled_1m/process_socialgraph/instances'
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python $PROJECT_ROOT/data/handle_xbox_feat_graph/reserved_edges_id_mapping.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --raw_data_root $RAW_DATA_ROOT \
