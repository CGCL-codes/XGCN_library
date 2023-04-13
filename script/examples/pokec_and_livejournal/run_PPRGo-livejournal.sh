all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data
config_file_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/xGCN/config

dataset=livejournal
model=PPRGo
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_whole_graph --val_batch_size 256 \
    --file_val_set $data_root/val_edges.pkl \
    --test_method multi_pos_whole_graph --test_batch_size 256 \
    --file_test_set $data_root/test_set.pkl \
    --from_pretrained 0 --file_pretrained_emb $file_pretrained_emb \
    --ppr_data_root $all_data_root/model_output/$dataset/PPR/[seed0] \
