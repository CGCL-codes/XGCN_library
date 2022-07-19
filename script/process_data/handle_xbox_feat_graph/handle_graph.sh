PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='xbox-feat-1m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
FILE_INPUT='/media/xreco/DEV/socgraph/usconsoleall_sampled_1m/process_socialgraph/instances/social_graph_past.tsv'

python $PROJECT_ROOT/data/handle_xbox_feat_graph/handle_graph.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --dataset_name $DATASET \
    --dataset_type 'social' \
    --file_input $FILE_INPUT \
