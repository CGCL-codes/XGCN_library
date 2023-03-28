all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data
config_file_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/xGCN/config

######################
dataset=pokec

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/UltraGCN/data

python -m XGCN.model.UltraGCN.prepare_UltraGCN_data \
    --data_root $data_root --results_root $results_root \
    --topk 10 \

######################
dataset=livejournal

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/UltraGCN/data

python -m XGCN.model.UltraGCN.prepare_UltraGCN_data \
    --data_root $data_root --results_root $results_root \
    --topk 10 \
