PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='pokec'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

EMB_RESULTS_DIR='pprgo/[new_ppr][lr0.005][bpr][reg0]'
EMB_RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$EMB_RESULTS_DIR
RESULTS_ROOT=$EMB_RESULTS_ROOT'/top100-reco'

# RESULTS_DIR='ppr/top100-reco'
# RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT/case_study/calc_hop.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input $RESULTS_ROOT'/top_reco.pkl' \
    --file_output $RESULTS_ROOT'/top_reco_hops.pkl' \
