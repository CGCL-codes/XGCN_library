all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=gowalla
model=ELPH
seed=0
device='cuda:1'
emb_table_device=$device
forward_device=$device
out_emb_table_device=$device

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed][epoch_sample_ratio1.0]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method multi_pos_whole_graph \
    --file_val_set $data_root/val.pkl \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test.pkl \
    --device $device \
    --epochs 1000 --val_freq 1 --convergence_threshold 100 \
    --key_score_metric r20 \
    --dnn_arch "[nn.Linear(self.get_input_dim(), 1024), nn.ReLU(), nn.Linear(1024, 1)]" \
    --max_hash_hops 2 \
    --minhash_num_perm 128 \
    --hll_p 8 \
