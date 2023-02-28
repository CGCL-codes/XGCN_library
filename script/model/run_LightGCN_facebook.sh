project_root="/media/xreco/DEV/xiran/code/gnn_zoo"
all_data_root="/media/xreco/DEV/xiran/data/gnn_zoo"

dataset="facebook"
model="LightGCN"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

graph_device="cuda:0"
emb_table_device="cuda:0"
gnn_device="cuda:0"
out_emb_table_device="cuda:0"

results_root=$all_data_root"/model_output/"$dataset"/"$model"/[seed$seed]"

python $project_root/main/main.py $project_root \
    --config_file $project_root"/model/"$model"/config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --num_gcn_layers 2 --train_num_layer_sample "[]" \
    --val_evaluator "WholeGraph_MultiPos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_set.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test_set.pkl" \
    --epochs 200 --val_freq 1 \
    --graph_device $graph_device --num_workers 0 \
    --emb_table_device $emb_table_device \
    --gnn_device $gnn_device \
    --out_emb_table_device $out_emb_table_device \
    --from_pretrained 0 --file_pretrained_emb "" \
    --freeze_emb 0 --use_sparse 0 \
    --emb_lr 0.01 \
    --loss_type "bpr" \
    --L2_reg_weight 1e-4 \
    --stack_layers 1 \
    --infer_num_layer_sample "[]" \
