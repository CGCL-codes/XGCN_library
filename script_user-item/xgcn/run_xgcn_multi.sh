PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cuda:0'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='yelp2018' ## 'yelp2018' 'gowalla' 'amazon-book'  'taobao-1.6m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1

T=9999  #5
K=0  #999999 
LOAD_BEST=1
USE_TWO_REFNET=0
L2_REG=1e-3
endure=6
num_gcn_layers=2

RESULTS_DIR="xgcn_multi/[bce]multi-pretrained/[ultragcn][bce][res][gcn_layers=$num_gcn_layers][use2refnet=$USE_TWO_REFNET][3layer-ffn][reg$L2_REG][T$T-K$K][load_best=$LOAD_BEST][endure=$endure]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

mkdir -p $RESULTS_ROOT
LOG_FILE=$RESULTS_ROOT'/log.txt'

echo "[begin at $(date)]" > $LOG_FILE
echo "results dir: "$RESULTS_ROOT

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'xgcn_multi' \
    --config_file $CONFIG_ROOT'/xgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --emb_table_device $DEVICE \
    --validation_method 'multi_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/test-1000.pkl' \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --key_score_metric 'r20' \
    --convergence_threshold 100 --epochs 1000 \
    --train_dl 'EdgeBased_Sampling_TrainDataLoader' --edge_sample_ratio 0.333 \
    --prop_type 'lightgcn' --num_gcn_layers $num_gcn_layers --stack 1 \
    --use_dnn_list 1 \
    --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
    --use_two_dnn $USE_TWO_REFNET \
    --renew_and_prop_freq $T --K $K --endure $endure \
    --renew_by_loading_best $LOAD_BEST \
    --emb_init_std 1.0 \
    --l2_reg_weight $L2_REG \
    --loss_fn 'bce_loss' --num_neg 32 --neg_weight 4 \
    --from_pretrained 1 --file_pretrained_emb '/media/xreco/DEV/xiran/data/social_and_user_item/model_outputs/gnn_yelp2018/ultragcn/[v0][0]/out_emb_table.pt' \
    >> $LOG_FILE
    # --from_pretrained 1 --file_pretrained_emb $ALL_RESULTS_ROOT'/gnn_'$DATASET'/node2vec/[0]/out_emb_table.pt' \

echo "[end at $(date)]" >> $LOG_FILE

# [torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]
# [torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]


# DATASET='gowalla' ## 'yelp2018' 'gowalla' 'amazon-book'  'taobao-1.6m'

# DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

# ################
# SEED=1

# T=15  #5
# K=3  #999999 
# LOAD_BEST=1
# USE_SNN=1
# USE_TWO_REFNET=0
# L2_REG=1e-3
# endure=9
# num_gcn_layers=3

# RESULTS_DIR="xgcn_multi/[bce][gcn_layers=$num_gcn_layers][1dnn][2layer-ffn][prop1][reg$L2_REG][T$T-K$K][load_best=$LOAD_BEST][USE_SNN=$USE_SNN][endure=$endure]"
# RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

# mkdir -p $RESULTS_ROOT
# LOG_FILE=$RESULTS_ROOT'/log.txt'

# echo "[begin at $(date)]" > $LOG_FILE
# echo "results dir: "$RESULTS_ROOT

# python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
#     --model 'xgcn_multi' \
#     --config_file $CONFIG_ROOT'/xgcn-config.yaml' \
#     --data_root $DATA_ROOT \
#     --results_root $RESULTS_ROOT \
#     --seed $SEED \
#     --device $DEVICE \
#     --emb_table_device $DEVICE \
#     --validation_method 'multi_pos_whole_graph' \
#     --mask_nei_when_validation 1 \
#     --file_validation $DATA_ROOT'/test-1000.pkl' \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test.pkl' \
#     --key_score_metric 'r20' \
#     --convergence_threshold 100 --epochs 1000 \
#     --train_dl 'EdgeBased_Sampling_TrainDataLoader' --edge_sample_ratio 0.333 \
#     --prop_type 'lightgcn' --num_gcn_layers $num_gcn_layers --stack 1 \
#     --use_special_dnn $USE_SNN --dnn_arch '[torch.nn.Linear(64*4, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
#     --scale_net_arch '[torch.nn.Linear(64*4, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1), torch.nn.Sigmoid()]' \
#     --use_two_dnn $USE_TWO_REFNET \
#     --renew_and_prop_freq $T --K $K --endure $endure \
#     --renew_by_loading_best $LOAD_BEST \
#     --emb_init_std 1.0 \
#     --l2_reg_weight $L2_REG \
#     --loss_fn 'bce_loss' --num_neg 32 --neg_weight 4 \
#     >> $LOG_FILE
#     # --from_pretrained 1 --file_pretrained_emb $ALL_RESULTS_ROOT'/gnn_'$DATASET'/node2vec/[0]/out_emb_table.pt' \

# echo "[end at $(date)]" >> $LOG_FILE

# # [torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]
# # [torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]
