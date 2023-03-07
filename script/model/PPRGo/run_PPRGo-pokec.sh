project_root="/media/xreco/DEV/xiran/code/XGCN"
all_data_root="/media/xreco/DEV/xiran/data/XGCN"

dataset="pokec"
model="PPRGo"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

ppr_data_device="cuda:0"
emb_table_device="cuda:0"
forward_device="cuda:0"
out_emb_table_device="cuda:0"

results_root=$all_data_root"/model_output/"$dataset"/"$model"/debug"

python $project_root/main/main.py $project_root \
    --config_file $project_root"/model/"$model"/config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --ppr_data_root $all_data_root"/model_output/"$dataset"/PPR/undirected-top100" \
    --val_evaluator "WholeGraph_OnePos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_edges-1000.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test.pkl" \
    --epochs 200 --val_freq 1 \
    --ppr_data_device $ppr_data_device \
    --emb_table_device $emb_table_device \
    --forward_device $forward_device \
    --out_emb_table_device $out_emb_table_device \
    --from_pretrained 0 --file_pretrained_emb "" \
    --freeze_emb 0 --use_sparse 0 \
    --emb_lr 0.01 \
    --topk 32 --use_uniform_weight 1 \
    --loss_type "bpr" \
    --L2_reg_weight 0.0 \
