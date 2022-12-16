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
num_layer_sample=$7

RESULTS_DIR="sgc_learnable_emb/[seed$SEED][gcn_layer$num_gcn_layers]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

file_pretrained_emb=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/node2vec/[best]/out_emb_table.pt'

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/sgc_learnable_emb-config.yaml' \
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
    --convergence_threshold 5 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \
    --num_gcn_layers $num_gcn_layers \
    --num_layer_sample $num_layer_sample \

find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
