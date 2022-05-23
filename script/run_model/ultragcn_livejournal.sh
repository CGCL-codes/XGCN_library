PROJECT_ROOT=$1
ALL_DATA_ROOT=$2

DEVICE=$3

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='livejournal'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

#### prepare_ultragcn_data 
RESULTS_DIR='ultragcn/data'
python $PROJECT_ROOT'/'model/UltraGCN/prepare_ultragcn_data.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 32 \

################
SEED=1
RESULTS_DIR='ultragcn/[best'$SEED']'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/ultragcn/ultragcn-config.yaml' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --file_ultra_constrain_mat $RESULTS_ROOT'/$PROJECT_ROOT'/'data/constrain_mat.pkl' \
    --file_ii_topk_neighbors $RESULTS_ROOT'/$PROJECT_ROOT'/'data/beta_score_topk/ii_topk_neighbors.np.pkl' \
    --file_ii_topk_similarity_scores $RESULTS_ROOT'/$PROJECT_ROOT'/'data/beta_score_topk/ii_topk_similarity_scores.np.pkl' \
    --device $DEVICE \
    --loss_fn 'bce_loss' \
    --emb_lr 0.005 \
    --num_neg 128 \
    --neg_weight 128 \
    --lambda 0.8 \
    --gamma 3.5 \
    --l2_reg_weight 1e-5 \
    --validation_method 'one_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/validation.pkl' \
    --key_score_metric 'r100' \
    --convergence_threshold 20 \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \

# find $RESULTS_ROOT -name "*.pt" -type f -print -exec rm -rf {} \;
