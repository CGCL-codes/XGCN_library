all_data_root=$1

#####################
dataset=pokec

data_root=$all_data_root/dataset/instance_$dataset

python -m XGCN.data.from_txt_to_np \
    --file_input $data_root/validation.txt \
    --file_output $data_root/val_edges.pkl \

python -m XGCN.data.from_txt_adj_to_adj_eval_set \
    --file_input $data_root/test.txt \
    --file_output $data_root/test_set.pkl \

#####################
dataset=livejournal

data_root=$all_data_root/dataset/instance_$dataset

python -m XGCN.data.from_txt_to_np \
    --file_input $data_root/validation.txt \
    --file_output $data_root/val_edges.pkl \

python -m XGCN.data.from_txt_adj_to_adj_eval_set \
    --file_input $data_root/test.txt \
    --file_output $data_root/test_set.pkl \
