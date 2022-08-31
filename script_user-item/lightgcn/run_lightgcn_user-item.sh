ALL_DATA_ROOT='/home/jialia/reco'
PROJECT_ROOT="$ALL_DATA_ROOT/xGCN"


DEVICE='cuda:0'   ##'cuda:0'  'cpu'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='amazon-book-1.5m'  ## 'taobao-1.6m' 'amazon-book-1.5m'
## on amazon-book-1.5m:  13GB GPU, 1 hour per epoch

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
RESULTS_DIR='lightgcn/[0]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'lightgcn' \
    --config_file $CONFIG_ROOT'/lightgcn-config.yaml' --use_sparse 0 \
    --data_root $DATA_ROOT \
    --seed $SEED \
    --results_root $RESULTS_ROOT \
    --device $DEVICE \
    --validation_method 'multi_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val-1000.pkl' \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --train_dl 'NodeBased_TrainDataLoader' --edge_sample_ratio 1.0 \
    --key_score_metric 'r20' \
    --convergence_threshold 50 --epochs 1000 --val_freq 5 \
    --train_batch_size 2048 \
    --emb_lr 0.001 \
    --num_gcn_layers 2 \
    --stack_layers 1 \
    --l2_reg_weight 1e-3 \
    --loss_fn 'bpr_loss' \