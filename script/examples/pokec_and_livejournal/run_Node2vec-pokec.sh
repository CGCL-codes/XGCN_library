all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data
config_file_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/xGCN/config

dataset=pokec
model=Node2vec
seed=0
device="cuda:0"

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_whole_graph --val_batch_size 256 \
    --file_val_set $data_root/val_edges.pkl \
    --test_method multi_pos_whole_graph --test_batch_size 256 \
    --file_test_set $data_root/test_set.pkl \
    --device $device \
    --p 1.0 --q 10.0 \
    --context_size 3 \
