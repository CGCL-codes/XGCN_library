all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data

#####################
dataset=pokec

data_root=$all_data_root/dataset/instance_$dataset

python -m XGCN.data.from_txt_to_csr \
    --file_input $data_root/train.txt \
    --results_root $data_root \
    --graph_type homo \
    --is_adj_list 0 \

#####################
dataset=livejournal

data_root=$all_data_root/dataset/instance_$dataset

python -m XGCN.data.from_txt_to_csr \
    --file_input $data_root/train.txt \
    --results_root $data_root \
    --graph_type homo \
    --is_adj_list 0 \
