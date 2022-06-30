PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='pokec'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

RESULTS_DIR='pprgo/[new_ppr][lr0.005][bpr][reg0]'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

TOPK=100

python $PROJECT_ROOT/case_study/get_emb_top_reco.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --file_input $RESULTS_ROOT'/out_emb_table.pt' \
    --file_output $RESULTS_ROOT'/top'$TOPK'-reco.pkl' \
    --num_sample 1000 \
    --topk $TOPK \
