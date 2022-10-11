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

num_gcn_layers=$6
reg=$7
use_dnn=$8

RESULTS_DIR="lightgcn/[num_gcn_layers$num_gcn_layers][reg$reg][use_dnn$use_dnn][seed$SEED]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --model 'block_lightgcn' --train_dl 'EdgeBased_Sampling_Block_TrainDataLoader' \
    --config_file $CONFIG_ROOT'/lightgcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --device $DEVICE \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 10 --train_batch_size 2048 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --num_gcn_layers $num_gcn_layers --num_layer_sample '[]' --num_workers 6 \
    --l2_reg_weight $reg \
    --stack_layers 0 \
    --loss_fn 'bpr_loss' --num_neg 1 \
    --use_special_dnn $use_dnn \
    --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
    --scale_net_arch '[torch.nn.Linear(64, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1), torch.nn.Sigmoid()]' \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
