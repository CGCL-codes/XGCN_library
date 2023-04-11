all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data
config_file_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/xGCN/config

dataset=facebook
model=UltraGCN
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]
ultragcn_data_root=$all_data_root/model_output/$dataset/UltraGCN/data

file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method MultiPosWholeGraph_Evaluator --val_batch_size 256 \
    --file_val_set $data_root/val_set.pkl \
    --test_method MultiPosWholeGraph_Evaluator --test_batch_size 256 \
    --file_test_set $data_root/test_set.pkl \
    --file_ultra_constrain_mat $ultragcn_data_root/constrain_mat.pkl \
    --file_ii_topk_neighbors $ultragcn_data_root/beta_score_topk/ii_topk_neighbors.np.pkl \
    --file_ii_topk_similarity_scores $ultragcn_data_root/beta_score_topk/ii_topk_similarity_scores.np.pkl \
