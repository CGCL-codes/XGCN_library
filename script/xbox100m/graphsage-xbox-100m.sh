PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cpu'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-100m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
N2V_EMB='/media/xreco/DEV/xiran/data/social_and_user_item/model_outputs/gnn_xbox-100m/node2vec/saved/out_emb_table.pt'

################
SEED=1
RESULTS_DIR='graphsage/[0]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/graphsage-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_k_neg' \
    --mask_nei_when_validation 0 \
    --file_validation $DATA_ROOT'/test-1-99.pkl' --key_score_metric 'n20'  \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges-5000.pkl' \
    --train_batch_size 1024 \
    --emb_dim 32 \
    --epochs 120 --convergence_threshold 10 \
    --edge_sample_ratio 0.01 \
    --from_pretrained 1 --file_pretrained_emb $N2V_EMB \
    --freeze_emb 1 \
    --use_sparse 0 \
    --num_gcn_layers 2 --num_layer_sample '[10, 20]' \
    --gnn_arch "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]" \
    # --num_gcn_layers 1 --num_layer_sample '[10]' \
    # --gnn_arch "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]" \
