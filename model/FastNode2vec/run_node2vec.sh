PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DATASET='xbox-100m'
# DATASET='pokec'
DATA_ROOT=$ALL_DATA_ROOT'/datasets/instance_'$DATASET

RESULTS_DIR='fast_n2v/[0]'
RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs/gnn_'$DATASET'/'$RESULTS_DIR


python run_node2vec.py $PROJECT_ROOT \
    --data_root $DATA_ROOT --results_root $RESULTS_ROOT \
    --emb_dim 32 \
    --epochs 1000000 \
    --emb_lr 0.01 \
    --num_walks 1 \
    --walk_length 8 \
    --context_size 5 \
    --p 1 --q 10 \
