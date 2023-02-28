# please modify these two paths, and you can run all the stuff here by a "bash facebook_data_process.sh"
project_root='../..'  # root of the code
all_data_root='/home/sxr/data/XGCN'  # specify a directory to place the data

raw_data_root=$all_data_root'/dataset/raw_facebook'

mkdir -p $raw_data_root
cp $project_root'/example_data/raw_facebook/facebook_combined.txt' $raw_data_root

dataset='facebook'

####### process raw txt file to generate csr graph #######
echo -e ">>> process raw txt file to generate csr graph\n"
file_txt_graph='facebook_combined.txt'

raw_data_root=$all_data_root'/dataset/raw_'$dataset'/'

file_input=$raw_data_root'/'$file_txt_graph
results_root=$raw_data_root'/csr'

python $project_root/data_process/from_txt_to_csr.py $project_root \
    --file_input $file_input \
    --results_root $results_root \
    --is_adj_list 0 \
    --graph_type 'homo' \


####### remove some edges from the raw graph as positive samples for model evaluation #######
echo -e "\n>>> remove some edges from the raw graph as positive samples for model evaluation\n"
raw_data_root=$all_data_root'/dataset/raw_'$dataset'/'

data_root=$raw_data_root'/csr'
results_root=$all_data_root'/dataset/instance_'$dataset

python $project_root/data_process/pos_edges_split.py $project_root \
    --data_root $data_root \
    --results_root $results_root \
    --seed 1999 \
    --num_sample 5000 \
    --min_src_out_degree 3 \
    --min_dst_in_degree 1 \


####### turn the edges for evaluation into adjacency list and make validation/test split #######
echo -e "\n>>> turn the edges for evaluation into adjacency list and make validation/test split\n"
data_root=$all_data_root'/dataset/instance_'$dataset

python $project_root/data_process/from_edges_to_adj_eval_set.py $project_root \
    --data_root $data_root \
    --num_validation 1000 \


####### run GraphSAGE #######
echo -e "\n>>> run GraphSAGE\n"
model="GraphSAGE"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

graph_device="cuda:0"  # when graph is on GPU, num_workers must be 0
emb_table_device="cuda:0"
gnn_device="cuda:0"
out_emb_table_device="cuda:0"

results_root=$all_data_root"/model_output/"$dataset"/"$model"/[seed$seed]"

python $project_root/main/main.py $project_root \
    --config_file $project_root"/model/"$model"/config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --num_gcn_layers 2 --train_num_layer_sample "[10, 10]" \
    --val_evaluator "WholeGraph_MultiPos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_set.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test_set.pkl" \
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
