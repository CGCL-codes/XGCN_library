PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='pokec'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

TOPK=100

EMB_RESULTS_DIR='pprgo/[new_ppr][lr0.005][bpr][reg0]'
EMB_RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$EMB_RESULTS_DIR
RESULTS_ROOT=$EMB_RESULTS_ROOT'/top'$TOPK'-reco'

python $PROJECT_ROOT/case_study/get_emb_top_reco.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input $EMB_RESULTS_ROOT'/out_emb_table.pt' \
    --results_root $RESULTS_ROOT \
    --num_sample 1000 \
    --topk $TOPK \
