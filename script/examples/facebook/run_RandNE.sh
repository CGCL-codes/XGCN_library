# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=facebook
model=RandNE
seed=0
device="cpu"

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
    --device $device \
    --use_lightgcn_coe 1 \
    --orthogonal 1 \
    --num_gcn_layers 3 \
    --alpha_list "[0, 1e-1, 1e-1, 1e-1]" \
