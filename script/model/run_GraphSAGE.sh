project_root="/media/xreco/DEV/xiran/code/gnn_zoo"
all_data_root="/media/xreco/DEV/xiran/data/gnn_zoo"

dataset="pokec"
model="GraphSAGE"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

graph_device="cuda:0"
emb_table_device="cuda:0"
gnn_device="cuda:0"
out_emb_table_device="cuda:0"

results_root=$all_data_root"/model_output/"$dataset"/"$model"/debug"

python $project_root/main/main.py $project_root \
    --config_file $project_root"/model/"$model"/config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --num_gcn_layers 2 --train_num_layer_sample "[10, 10]" \
    --val_evaluator "WholeGraph_OnePos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_edges-1000.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test.pkl" \
    --epochs 200 --val_freq 1 \
    --forward_mode "sample" \
    --graph_device $graph_device --num_workers 0 \
    --emb_table_device $emb_table_device \
    --gnn_device $gnn_device \
    --out_emb_table_device $out_emb_table_device \
    --from_pretrained 0 --file_pretrained_emb "" \
    --freeze_emb 0 --use_sparse 0 \
    --emb_lr 0.01 \
    --gnn_arch "[{'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool', 'activation': torch.tanh}, {'in_feats': 64, 'out_feats': 64, 'aggregator_type': 'pool'}]" \
    --gnn_lr 0.01 \
    --loss_type "bpr" \
    --L2_reg_weight 1e-4 \
    --infer_num_layer_sample "[]" \


# python $project_root/main/main.py $project_root \
#     $project_root"/model/"$model"/config.yaml" \
#     "seed::${seed}" \
#     "model::str::${model}" \
#     "data_root::str::${data_root}" "results_root::str::${results_root}" \
#     "Dataset_type::str::BlockDataset" \
#     "train_num_layer_sample::[10, 10]" "num_gcn_layers::2" \
#     "NodeListDataset_type::str::LinkDataset" \
#     "pos_sampler_config|Sampler_type::str::ObservedEdges_Sampler" \
#     "neg_sampler_config|Sampler_type::str::RandomNeg_Sampler" \
#     "BatchSampleIndicesGenerator_type::str::SampleIndicesWithReplacement" \
#     "train_batch_size::2048" "train_edge_sample_ratio::0.1" \
#     "val_evaluator|Evaluator_type::str::WholeGraph_OnePos_Evaluator" "val_evaluator|batch_size::256" \
#     "val_evaluator|file_eval_set::str::${data_root}/val_edges-1000.pkl" \
#     "test_evaluator|Evaluator_type::str::WholeGraph_OnePos_Evaluator" "test_evaluator|batch_size::256" \
#     "test_evaluator|file_eval_set::str::${data_root}/test.pkl" \
#     "epochs::200" "val_freq::1" \
#     "key_score_metric::str::r100" "convergence_threshold::20" \
#     "forward_mode::str::sample" \
#     "graph_device::str::${graph_device}" \
#     "emb_table_device::str::${emb_table_device}" \
#     "gnn_device::str::${gnn_device}" \
#     "out_emb_table_device::str::${out_emb_table_device}" \
#     "from_pretrained::False" "file_pretrained_emb::None" \
#     "freeze_emb::False" "use_sparse::False" \
#     "emb_dim::64" "emb_init_std::0.1" "emb_lr::0.01" \
#     "gnn_arch::[{"in_feats": 64, "out_feats": 64, "aggregator_type": "pool", "activation": torch.tanh}, {"in_feats": 64, "out_feats": 64, "aggregator_type": "pool"}]" \
#     "gnn_lr::0.01" \
#     "loss_type::str::bpr" \
#     "L2_reg_weight::1e-5" \
#     "infer_num_layer_sample::[]" \
