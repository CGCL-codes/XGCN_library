# Script for xGCN on the amazon-book dataset

# The results of the following running should be around:
# r20:0.0452 || r50:0.0844 || r100:0.1302 || r300:0.2398 || n20:0.0355 || n50:0.0501 || n100:0.0650 || n300:0.0951
# 'r' for 'Recall@', 'n' for 'NDCG@'

# set to your own path:
all_data_root='/home/sxr/code/XGCN_library/XGCN_data'
config_file_root='/home/sxr/code/XGCN_library/config'

dataset=amazon-book
model=xGCN
seed=0
device='cuda:0'
emb_table_device=$device
forward_device=$device
out_emb_table_device=$device

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed][epoch_sample_ratio1.0]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method multi_pos_whole_graph \
    --file_val_set $data_root/val.pkl \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test.pkl \
    --emb_table_device $emb_table_device \
    --forward_device $forward_device \
    --out_emb_table_device $out_emb_table_device \
    --epochs 1000 --val_freq 1 --convergence_threshold 100 \
    --key_score_metric r20 \
    --epoch_sample_ratio 1.0 \
    --dnn_arch "[nn.Linear(64, 1024), nn.Tanh(), nn.Linear(1024, 64)]" \
    --use_scale_net 0 \
    --L2_reg_weight 1e-4 \
    --num_gcn_layers 1 \
    --stack_layers 1 \
    --renew_by_loading_best 1 \
    --T 5 \
    --K 99999 \
    --tolerance 5 \
