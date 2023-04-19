# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=facebook
model=GensimNode2vec
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_k_neg \
    --file_val_set $data_root/val-one_pos_k_neg.pkl \
    --key_score_metric r20 \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
