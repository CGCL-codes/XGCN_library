PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/_model_outputs'

DATASET='pokec'

RESULTS_DIR='n2v_best'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'case_study/emb_simple_stat.py $PROJECT_ROOT \
    --results_root $RESULTS_ROOT \


RESULTS_DIR='plot/dnn-lightgcn-final_version/[gcn1layer][scale][3layer][freq3][K10][endure3]'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'case_study/emb_simple_stat.py $PROJECT_ROOT \
    --results_root $RESULTS_ROOT \


RESULTS_DIR='plot/pprgo/[best][bpr][reg0]'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'case_study/emb_simple_stat.py $PROJECT_ROOT \
    --results_root $RESULTS_ROOT \
