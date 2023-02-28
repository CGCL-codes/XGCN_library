project_root="/home/sxr/code/XGCN"
all_data_root="/home/sxr/data/XGCN"

dataset="pokec"
model="xGCN"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

emb_table_device="cuda:0"
forward_device="cuda:0"
out_emb_table_device="cuda:0"

results_root=$all_data_root"/model_output/"$dataset"/"$model"/[seed$seed]"

python $project_root/main/main.py $project_root \
    --config_file $project_root"/model/"$model"/config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --val_evaluator "WholeGraph_OnePos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_edges-1000.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test.pkl" \
    --epochs 150 --val_freq 1 \
    --emb_table_device $emb_table_device \
    --forward_device $forward_device \
    --out_emb_table_device $out_emb_table_device \
    --from_pretrained 0 --file_pretrained_emb "" \
    --emb_dim 64 --emb_init_std 1.0 \
    --num_gcn_layers 1 \
    --L2_reg_weight 0.0 \
    --dnn_arch "[nn.Linear(64, 1024), nn.Tanh(), nn.Linear(1024, 64)]" \
    --use_scale_net 1 --scale_net_arch "[nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1), nn.Sigmoid()]" \
    --renew_by_loading_best 1 \
    --K 10 --T 3 --tolerance 3 \
