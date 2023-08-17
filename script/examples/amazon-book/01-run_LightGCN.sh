# Script to reproduce the LightGCN results on the amazon-book dataset

# The results of the following running should be around:
# r20:0.0409 || r50:0.0792 || r100:0.1252 || r300:0.2367 || n20:0.0316 || n50:0.0458 || n100:0.0606 || n300:0.0911
# 'r' for 'Recall@', 'n' for 'NDCG@'

# set to your own path:
all_data_root='/home/sxr/code/XGCN_library/XGCN_data'
config_file_root='/home/sxr/code/XGCN_library/config'

dataset=amazon-book
model=LightGCN
seed=0
device="cuda:0"
graph_device=$device
emb_table_device=$device
gnn_device=$device
out_emb_table_device=$device

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

# In LightGCN's official code (https://github.com/gusye1234/LightGCN-PyTorch), 
# for each epoch, there are num_edges samples. For each sample, firstly, a user 
# is randomly sampled. Then a neighbor (item) of the user is sampled as the positive node. 

# The amazon-book dataset has 52643 users and 2380730 interactions (edges). 
# 2380730 / 52643 = 45.22
# To reproduce the LightGCN's setting, in XGCN, we use the 
# NodeBased_ObservedEdges_Sampler, and set:
# str_num_total_samples=num_users
# epoch_sample_ratio=45.22

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-full_graph-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method multi_pos_whole_graph \
    --file_val_set $data_root/test.pkl \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test.pkl \
    --str_num_total_samples num_users \
    --pos_sampler NodeBased_ObservedEdges_Sampler \
    --neg_sampler StrictNeg_Sampler \
    --epoch_sample_ratio 45.22 \
    --num_gcn_layers 2 \
    --L2_reg_weight 1e-4 --use_ego_emb_L2_reg 1 \
    --emb_lr 0.001 \
    --emb_dim 64 \
    --train_batch_size 2048 \
    --epochs 1000 --val_freq 20 \
    --key_score_metric r20 --convergence_threshold 100 \
    --graph_device $graph_device --emb_table_device $emb_table_device \
    --gnn_device $gnn_device --out_emb_table_device $out_emb_table_device \
