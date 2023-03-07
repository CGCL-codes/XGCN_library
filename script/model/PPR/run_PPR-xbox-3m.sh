all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data
config_file_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/xGCN/config

dataset=xbox-3m
model=PPR
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

python -m XGCN.main.run_ppr \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --seed $seed \
