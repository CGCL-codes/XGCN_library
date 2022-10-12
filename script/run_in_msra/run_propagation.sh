PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$3

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1

prop_times=$4

RESULTS_DIR="propagation/[prop${prop_times}]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/run_propagation.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --emb_dim 64 --emb_init_std 1.0 \
    --prop_type 'lightgcn' --num_gcn_layers 1 --stack_layers 0 --use_numba_csr_mult 0 \
    --prop_times $prop_times \

python $PROJECT_ROOT/main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/val_edges-1000.pkl' \
    --file_output $RESULTS_ROOT'/val.json' \

python $PROJECT_ROOT/main/eval.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --file_output $RESULTS_ROOT'/test.json' \
