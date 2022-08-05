PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cuda:0'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='gowalla'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
RESULTS_DIR='xgcn/[0]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR
T=2
K=10

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
    --validation_method 'multi_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/test-1000.pkl' \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --key_score_metric 'r20' \
    --convergence_threshold 50 \
    --train_dl 'EdgeBased_Full_TrainDataLoader' \
    --prop_type 'lightgcn' --num_gcn_layers 1 --stack 1 \
    --use_special_dnn 1 --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
    --renew_and_prop_freq $T --K $K --endure 3 \
    --emb_init_std 1.0 \
    # >> $LOG_FILE

echo "[end at $(date)]" >> $LOG_FILE
