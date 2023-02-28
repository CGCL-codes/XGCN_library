project_root="/media/xreco/DEV/xiran/code/XGCN"
all_data_root="/media/xreco/DEV/xiran/data/XGCN"

config_root=$project_root'/XGCN/config'
all_dataset_root=$all_data_root'/dataset'
all_results_root=$all_data_root'/model_output'

dataset='pokec'

data_root=$all_dataset_root'/instance_'$dataset

################
seed=1

model='UltraGCN'
results_dir=$model/"[$seed]"
results_root=$all_results_root'/gnn_'$dataset'/'$results_dir

python $project_root/XGCN/main/run_model/run_model.py \
    --config_file $project_root/XGCN"/model/"$model/"config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --results_root $results_root \
    --val_evaluator "WholeGraph_OnePos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_edges-1000.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test.pkl" \
    --epochs 200 --val_freq 1 --key_score_metric 'r100' --convergence_threshold 10 \
    --file_ultra_constrain_mat $results_root'/../data/constrain_mat.pkl' \
    --file_ii_topk_neighbors $results_root'/../data/beta_score_topk/ii_topk_neighbors.np.pkl' \
    --file_ii_topk_similarity_scores $results_root'/../data/beta_score_topk/ii_topk_similarity_scores.np.pkl' \
    --device 'cuda' --emb_table_device 'cuda' \
    --loss_fn 'bce_loss' \
    --emb_lr 0.005 \
    --num_neg 128 \
    --neg_weight 128 \
    --lambda 0.8 \
    --gamma 3.5 \
    --L2_reg_weight 0.0 \

# find $results_root -name "*.pt" -type f -print -exec rm -rf {} \;
