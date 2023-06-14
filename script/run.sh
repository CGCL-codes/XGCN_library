all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=$1
model=ELPH
seed=0
max_hash_hops=1
device='cuda:1'

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed][CN]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method multi_pos_whole_graph \
    --file_val_set $data_root/test_set.pkl \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test_set.pkl \
    --device $device \
    --epochs 1 --val_freq 1 --convergence_threshold 10 \
    --key_score_metric r20 \
    --max_hash_hops $max_hash_hops \
    --minhash_num_perm 128 \
    --hll_p 8 \
    --dnn_arch "[nn.Linear(self.get_input_dim(), 1024), nn.ReLU(), nn.Linear(1024, 1)]" \
    
    # --p_drop 0.0 \
