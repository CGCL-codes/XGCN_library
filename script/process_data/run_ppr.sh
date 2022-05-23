PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$3
RESULTS_DIR='ppr/undirected-top100'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'model/PPR/run_ppr.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/ppr-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
