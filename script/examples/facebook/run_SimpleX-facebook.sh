all_data_root=/home/sxr/code/XGCN_and_data/data
config_file_root=/home/sxr/code/XGCN_and_data/xGCN/config

dataset=ali_ifashion-160k
model=SimpleX
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_evaluator WholeGraph_MultiPos_Evaluator --val_batch_size 256 \
    --file_val_set $data_root/val_edges.pkl \
    --test_evaluator WholeGraph_MultiPos_Evaluator --test_batch_size 256 \
    --file_test_set $data_root/test_edges.pkl \
