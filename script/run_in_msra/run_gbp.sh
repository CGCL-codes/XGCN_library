PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$4

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=$5
walk_length=$6
alpha=$7
rmax_ratio=$8

RESULTS_DIR="gbp/best/[seed$SEED][2layer_dnn][L$walk_length][alpha$alpha][rmax_ratio$rmax_ratio]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

file_pretrained_emb=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/node2vec/[best]/out_emb_table.pt'

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/gbp-config.yaml' \
    --data_root $DATA_ROOT \
    --seed $SEED \
    --results_root $RESULTS_ROOT \
    --device $DEVICE \
    --train_batch_size 1024 \
    --l2_reg_weight 0.0 \
    --loss_fn 'bpr_loss' \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 20 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --file_pretrained_emb $file_pretrained_emb \
    --walk_length $walk_length \
    --alpha $alpha \
    --rmax_ratio $rmax_ratio \
    --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \

find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;


# RESULTS_DIR="gbp/[seed$SEED][3layer_dnn][L$walk_length][alpha$alpha][rmax_ratio$rmax_ratio]"
# RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

# file_pretrained_emb=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/node2vec/[best]/out_emb_table.pt'

# python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
#     --config_file $CONFIG_ROOT'/gbp-config.yaml' \
#     --data_root $DATA_ROOT \
#     --seed $SEED \
#     --results_root $RESULTS_ROOT \
#     --device $DEVICE \
#     --train_batch_size 1024 \
#     --l2_reg_weight 0.0 \
#     --loss_fn 'bpr_loss' \
#     --validation_method 'one_pos_whole_graph' \
#     --mask_nei_when_validation 1 \
#     --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
#     --key_score_metric 'r100' \
#     --convergence_threshold 20 \
#     --test_method 'multi_pos_whole_graph' \
#     --mask_nei_when_test 1 \
#     --file_test $DATA_ROOT'/test.pkl' \
#     --file_pretrained_emb $file_pretrained_emb \
#     --walk_length $walk_length \
#     --alpha $alpha \
#     --rmax_ratio $rmax_ratio \
#     --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
