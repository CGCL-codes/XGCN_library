PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-3m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
N2V_EMB=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/n2v_best/out_emb_table.pt'

################
SEED=1
RESULTS_DIR='gin/[best'$SEED']'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/gin-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/validation.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 10 --epochs 200 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --from_pretrained 1 --file_pretrained_emb $N2V_EMB \
    --freeze_emb 1 \
    --num_gcn_layers 1 --num_layer_sample '[20]' \
    # --num_gcn_layers 2 --num_layer_sample '[10, 20]' \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
