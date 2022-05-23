PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='pokec'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
N2V_EMB=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/node2vec/[best1]/out_emb_table.pt'

################
SEED=1
RESULTS_DIR='gat/[best'$SEED']'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/gat-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 10 --epochs 200 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --from_pretrained 1 --file_pretrained_emb $N2V_EMB \
    --freeze_emb 0 \
    --num_gcn_layers 2 --num_layer_sample '[10, 20]' \
    --gnn_arch "[{'in_feats': 64, 'out_feats': 64, 'num_heads': 4, 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'num_heads': 4}]" \
    # --num_gcn_layers 1 --num_layer_sample '[20]' \
    # --gnn_arch "[{'in_feats': 64, 'out_feats': 64, 'num_heads': 4}]" \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
