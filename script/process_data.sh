PROJECT_ROOT='/home/xxx/code/xgcn'
ALL_DATA_ROOT='/home/xxx/data/xgcn_data'

cd process_data

DATASET='pokec'

bash handle_edge_list_txt.sh $PROJECT_ROOT $ALL_DATA_ROOT $DATASET

bash handle_src_pos_neg_eval_set.sh $PROJECT_ROOT $ALL_DATA_ROOT $DATASET

bash handle_adj_eval_set.sh $PROJECT_ROOT $ALL_DATA_ROOT $DATASET

# bash run_ppr.sh $PROJECT_ROOT $ALL_DATA_ROOT $DATASET

# bash prepare_ultragcn_data.sh $PROJECT_ROOT $ALL_DATA_ROOT $DATASET
