PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cpu'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-100m'

#########################
######## run ppr ########

RESULTS_DIR='ppr/undirected-top32'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'model/PPR/run_ppr.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/ppr-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --topk 32 \

###########################
######## run pprgo ########

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET
PPR_DATA_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/ppr/undirected-top32'

SEED=1
RESULTS_DIR='pprgo/[test]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/pprgo-config.yaml' \
    --data_root $DATA_ROOT \
    --seed $SEED \
    --ppr_data_root $PPR_DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --device $DEVICE \
    --emb_lr 0.02 \
    --l2_reg_weight 0.0 \
    --loss_fn 'bpr_loss' \
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
    --use_sparse 1 \
    # --from_pretrained 1 --file_pretrained_emb $RESULTS_ROOT'/base_emb_table.pt'


### for debug on pokec:
# python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
#     --config_file $CONFIG_ROOT'/pprgo-config.yaml' \
#     --data_root $DATA_ROOT \
#     --seed $SEED \
#     --ppr_data_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/ppr/undirected-top100' \
#     --results_root $RESULTS_ROOT \
#     --device $DEVICE \
#     --emb_lr 0.005 \
#     --l2_reg_weight 0.0 \
#     --loss_fn 'bpr_loss' \
#     --validation_method 'one_pos_whole_graph' \
#     --mask_nei_when_validation 1 \
#     --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
#     --key_score_metric 'r100' \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test.pkl' \
#     --train_batch_size 1024 \
#     --emb_dim 32 \
#     --epochs 120 --convergence_threshold 10 \
#     --edge_sample_ratio 0.01 \
#     --use_sparse 1 \
#     --from_pretrained 1 --file_pretrained_emb $RESULTS_ROOT'/base_emb_table.pt' \
