# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=facebook
model=xGCN
seed=0
device='cuda:0'
emb_table_device=$device
forward_device=$device
out_emb_table_device=$device

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

# file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/model/out_emb_table.pt

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_k_neg \
    --file_val_set $data_root/val-one_pos_k_neg.pkl \
    --key_score_metric r20 \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
    --emb_table_device $emb_table_device \
    --forward_device $forward_device \
    --out_emb_table_device $out_emb_table_device \
    --use_scale_net 0 \
    # --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \
