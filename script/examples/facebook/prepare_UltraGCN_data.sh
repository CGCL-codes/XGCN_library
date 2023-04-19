# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=facebook

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/UltraGCN/data

python -m XGCN.model.UltraGCN.prepare_UltraGCN_data \
    --data_root $data_root --results_root $results_root \
    --topk 10 \
