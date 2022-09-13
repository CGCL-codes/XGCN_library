PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

######################
DATASET=$4
## 'yelp2018' 'gowalla' 'amazon-book'  'taobao-1.6m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1

T=2  #5
K=50  #999999 
LOAD_BEST=1
USE_SNN=1
USE_TWO_REFNET=0
L2_REG=$5
# dnn_lr=0
endure=5

gamma=$6
topk=$7

RESULTS_DIR="xgcn/gamma/[test][topk=$topk][gamma=$gamma][reg=$L2_REG]"
# RESULTS_DIR="xgcn/case_study/[prop_only]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

mkdir -p $RESULTS_ROOT
LOG_FILE=$RESULTS_ROOT'/log.txt'

echo "[begin at $(date)]" > $LOG_FILE
echo "results dir: "$RESULTS_ROOT

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'xgcn' \
    --config_file $CONFIG_ROOT'/xgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --emb_table_device $DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 50 --epochs 500 \
    --train_dl 'EdgeBased_Sampling_TrainDataLoader' \
    --prop_type 'lightgcn' --num_gcn_layers 1 --stack 1 \
    --use_special_dnn $USE_SNN --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
    --scale_net_arch '[torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1), torch.nn.Sigmoid()]' \
    --use_two_dnn $USE_TWO_REFNET \
    --renew_and_prop_freq $T --K $K --endure $endure \
    --renew_by_loading_best $LOAD_BEST \
    --emb_init_std 1.0 \
    --l2_reg_weight $L2_REG \
    --file_ii_topk_neighbors $ALL_RESULTS_ROOT'/gnn_'$DATASET'/ultragcn/data/beta_score_topk/ii_topk_neighbors.np.pkl' \
    --file_ii_topk_similarity_scores $ALL_RESULTS_ROOT'/gnn_'$DATASET'/ultragcn/data/beta_score_topk/ii_topk_similarity_scores.np.pkl' \
    --topk $topk --gamma $gamma \
    # >> $LOG_FILE

echo "[end at $(date)]" >> $LOG_FILE

find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -r {} \;

# [torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]
# [torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]
