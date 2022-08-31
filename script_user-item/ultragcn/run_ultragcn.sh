PROJECT_ROOT='/home/songxiran/code/xGCN'
ALL_DATA_ROOT='/home/songxiran/data/social_and_user_item'

DEVICE=$1

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET=$2

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

#### prepare_ultragcn_data 
RESULTS_DIR='ultragcn/data'
python $PROJECT_ROOT'/'model/UltraGCN/prepare_ultragcn_data.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 100 \

################
SEED=1

lambda=$3
gamma=$4
reg=$5
num_neg=$6
neg_weight=$7
topk=$8

RESULTS_DIR="ultragcn/[lambda$lambda][gamma$gamma][reg$reg][neg$num_neg-weight$neg_weight][K$topk]"

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/ultragcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --file_ultra_constrain_mat $RESULTS_ROOT'/../data/constrain_mat.pkl' \
    --file_ii_topk_neighbors $RESULTS_ROOT'/../data/beta_score_topk/ii_topk_neighbors.np.pkl' \
    --file_ii_topk_similarity_scores $RESULTS_ROOT'/../data/beta_score_topk/ii_topk_similarity_scores.np.pkl' \
    --device $DEVICE \
    --loss_fn 'bce_loss' \
    --train_batch_size 2048 \
    --emb_lr 0.005 \
    --num_neg $num_neg \
    --neg_weight $neg_weight \
    --lambda $lambda \
    --gamma $gamma \
    --l2_reg_weight $reg \
    --topk $topk \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/val_edges-1000.pkl' \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 30 --epochs 500 \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
