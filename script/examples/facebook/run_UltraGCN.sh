# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=facebook
model=UltraGCN
seed=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]
ultragcn_data_root=$all_data_root/model_output/$dataset/UltraGCN/data

# file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/model/out_emb_table.pt

python -m XGCN.main.run_model --seed $seed \
    --config_file $config_file_root/$model-config.yaml \
    --data_root $data_root --results_root $results_root \
    --val_method one_pos_k_neg \
    --file_val_set $data_root/val-one_pos_k_neg.pkl \
    --key_score_metric r20 \
    --test_method multi_pos_whole_graph \
    --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
    --file_ultra_constrain_mat $ultragcn_data_root/constrain_mat.pkl \
    --file_ii_topk_neighbors $ultragcn_data_root/beta_score_topk/ii_topk_neighbors.np.pkl \
    --file_ii_topk_similarity_scores $ultragcn_data_root/beta_score_topk/ii_topk_similarity_scores.np.pkl \
    # --from_pretrained 1 --file_pretrained_emb $file_pretrained_emb \
