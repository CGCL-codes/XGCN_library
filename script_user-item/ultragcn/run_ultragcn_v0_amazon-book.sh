PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cuda:0'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='amazon-book'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

#### prepare_ultragcn_data 
RESULTS_DIR='ultragcn/data'
python $PROJECT_ROOT'/'model/UltraGCN/prepare_ultragcn_data.py $PROJECT_ROOT \
    --data_root $ALL_DATASETS_ROOT'/instance_'$DATASET \
    --results_root $ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 100 \

################
SEED=1
RESULTS_DIR='ultragcn/[v0][0]'

RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

python $PROJECT_ROOT'/'main/main.py $PROJECT_ROOT \
    --config_file $CONFIG_ROOT'/ultragcn-config.yaml' \
    --model 'ultragcn_v0' \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
    --seed $SEED \
    --file_ultra_constrain_mat $RESULTS_ROOT'/../data/constrain_mat.pkl' \
    --file_ii_topk_neighbors $RESULTS_ROOT'/../data/beta_score_topk/ii_topk_neighbors.np.pkl' \
    --file_ii_topk_similarity_scores $RESULTS_ROOT'/../data/beta_score_topk/ii_topk_similarity_scores.np.pkl' \
    --device $DEVICE \
    --train_dl 'EdgeBased_Full_TrainDataLoader' \
    --loss_fn 'bce_loss' \
    --emb_init_std 1e-4 \
    --emb_lr 1e-3 \
    --train_batch_size 1024 \
    --num_neg 500 \
    --neg_weight 500 \
    --topk 10 \
    --w1 1e-8 --w2 1 --w3 1 --w4 1e-8 \
    --gamma 2.75 \
    --l2_reg_weight 1e-4  \
    --validation_method 'multi_pos_whole_graph' \
    --mask_nei_when_validation 1 \
    --file_validation $DATA_ROOT'/test.pkl' \
    --test_method 'multi_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test.pkl' \
    --key_score_metric 'r20' \
    --convergence_threshold 15 --val_freq 5 \
    --epochs 2000 \
