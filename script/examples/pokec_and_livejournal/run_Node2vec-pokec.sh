all_data_root=$1
config_file_root=$2

dataset=pokec
model=Node2vec
seed=0
device="cuda:0"

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_whole_graph --file_val_set $data_root/val_edges.pkl \
    --test_method multi_pos_whole_graph --file_test_set $data_root/test_set.pkl \
    --device $device \
    --p 1.0 --q 10.0 \
    --context_size 3 \
