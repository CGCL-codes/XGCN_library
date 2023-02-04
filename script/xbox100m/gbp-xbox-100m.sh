PROJECT_ROOT='/media/xreco/jianxun/xGCN'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

DEVICE='cpu'

CONFIG_ROOT=$PROJECT_ROOT'/config'
ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

DATASET='xbox-100m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

################
SEED=1
walk_length=2
alpha=0.1
rmax_ratio=0.01

RESULTS_DIR="gbp/[seed$SEED][2layer_dnn][L$walk_length][alpha$alpha][rmax_ratio$rmax_ratio]"
RESULTS_ROOT=$ALL_RESULTS_ROOT'/gnn_'$DATASET'/'$RESULTS_DIR

N2V_EMB='/media/xreco/DEV/xiran/data/social_and_user_item/model_outputs/gnn_xbox-100m/fast_n2v/[0]/out_emb_table.pt'
file_pretrained_emb=$N2V_EMB

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
    --file_validation $DATA_ROOT'/test_edges-5000.pkl' --key_score_metric 'n20'  \
    --test_method 'one_pos_whole_graph' \
    --mask_nei_when_test 1 \
    --file_test $DATA_ROOT'/test_edges-5000.pkl' \
    --train_batch_size 1024 \
    --emb_dim 32 \
    --epochs 9999 --convergence_threshold 20 \
    --edge_sample_ratio 0.01 \
    --file_pretrained_emb $file_pretrained_emb \
    --walk_length $walk_length \
    --alpha $alpha \
    --rmax_ratio $rmax_ratio \
    --dnn_arch '[torch.nn.Linear(64, 1024), torch.nn.Tanh(), torch.nn.Linear(1024, 64)]' \
