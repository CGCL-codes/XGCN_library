all_data_root=$1
config_file_root=$2

dataset=pokec
model=UltraGCN
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]
ultragcn_data_root=$all_data_root/model_output/$dataset/UltraGCN/data

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_whole_graph --file_val_set $data_root/val_edges.pkl \
    --test_method multi_pos_whole_graph --file_test_set $data_root/test_set.pkl \
    --file_ultra_constrain_mat $ultragcn_data_root/constrain_mat.pkl \
    --file_ii_topk_neighbors $ultragcn_data_root/beta_score_topk/ii_topk_neighbors.np.pkl \
    --file_ii_topk_similarity_scores $ultragcn_data_root/beta_score_topk/ii_topk_similarity_scores.np.pkl \
    --num_neg 128 \
    --neg_weight 128 \
    --lambda 0.8 \
    --gamma 3.5 \
    --L2_reg_weight 0.0 \
