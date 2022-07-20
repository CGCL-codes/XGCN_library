PROJECT_ROOT='/media/xreco/jianxun/xGCN'
# ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DEVICE='cuda:0'
EMB_TABLE_DEVICE='cpu'

#########################################
DATASET='xbox-100m'
SEED=2022
T=2
K=10

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

RESULTS_DIR='xgcn/[0]'
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

mkdir -p $RESULTS_ROOT
LOG_FILE=$RESULTS_ROOT'/log.txt'

echo "[begin at $(date)]" > $LOG_FILE
echo "results dir: "$RESULTS_ROOT

# if the memory is enough, set --use_numba_csr_mult 0

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'xgcn' \
    --config_file $CONFIG_ROOT'/xgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --emb_table_device $EMB_TABLE_DEVICE \
    --validation_method 'one_pos_k_neg' \
    --mask_nei_when_validation 0 \
    --file_validation $DATA_ROOT'/test-1-99.pkl' \
    --test_method 'one_pos_k_neg' \
    --mask_nei_when_test 0 \
    --file_test $DATA_ROOT'/test-1-99.pkl' \
    --prop_type 'lightgcn' --num_gcn_layers 1 --use_numba_csr_mult 1 \
    --use_special_dnn 1 \
    --dnn_arch '[torch.nn.Linear(32, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 32)]' \
    --scale_net_arch '[torch.nn.Linear(32, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1), torch.nn.Sigmoid()]' \
    --renew_and_prop_freq $T --K $K --endure 1 \
    --emb_dim 32 \
    --emb_init_std 1.0 \
    --epochs 120 --convergence_threshold 4 \
    --edge_sample_ratio 0.01 \
    # >> $LOG_FILE

#   --validation_method 'one_pos_k_neg' \
#   --mask_nei_when_validation 0 \
#   --file_validation $DATA_ROOT'/test-1-99.pkl' \
#   --test_method 'one_pos_k_neg' \
#   --mask_nei_when_test 0 \
#   --file_test $DATA_ROOT'/test-1-99.pkl' \

# --validation_method 'one_pos_whole_graph' \
# --mask_nei_when_validation 1 \
# --val_batch_size 32 \
# --file_validation $DATA_ROOT'/test_edges-1000.pkl' \
# --test_method 'one_pos_whole_graph' \
# --mask_nei_when_test 1 \
# --test_batch_size 32 \
# --file_test $DATA_ROOT'/test_edges-1000.pkl' \

echo "[end at $(date)]" >> $LOG_FILE

# #########################################
# # save as .npy
# FILE_INPUT=$RESULTS_ROOT'/out_emb_table.pt'
# FILE_OUTPUT=$RESULTS_ROOT'/out_emb_table.npy'
# python $PROJECT_ROOT'/'data/save_pt_as_npy.py $PROJECT_ROOT \
#     --file_input $FILE_INPUT \
#     --file_output $FILE_OUTPUT \
