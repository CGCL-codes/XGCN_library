# input text file:
file_input_graph='/home/sxr/code/XGCN_and_data/XGCN_data/dataset/raw_gowalla/train.txt'
instance_root='/home/sxr/code/XGCN_and_data/XGCN_data/dataset/instance_gowalla2'

rm -r $instance_root
mkdir -p $instance_root

file_train_graph=$instance_root'/train.txt'

file_output_graph=$file_train_graph
seed=0

# available graph type: 'homo', 'user-item'
graph_type='user-item'
# available graph format: 'edge_list', 'adjacency_list'
graph_format='adjacency_list'

eval_method='one_pos_k_neg'
num_edge_samples=1000
num_neg=999
file_output_eval_set=$instance_root"/one_pos_k_neg-${num_edge_samples}-${num_neg}.txt"

min_src_out_degree=3
min_dst_in_degree=3

python -m XGCN.data.process.evaluation_set_generation \
    --file_input_graph $file_input_graph \
    --file_output_graph $file_output_graph \
    --file_output_eval_set $file_output_eval_set \
    --seed $seed --graph_type $graph_type --graph_format $graph_format \
    --num_edge_samples $num_edge_samples \
    --min_src_out_degree $min_src_out_degree \
    --min_dst_in_degree $min_dst_in_degree \
    --eval_method $eval_method \
    --num_neg $num_neg \

##########################
file_input_graph=$file_train_graph
file_output_graph=$file_train_graph
seed=1

# available graph type: 'homo', 'user-item'
graph_type='user-item'
# available graph format: 'edge_list', 'adjacency_list'
graph_format='edge_list'

eval_method='one_pos_whole_graph'
num_edge_samples=1000
# num_neg=1000
file_output_eval_set=$instance_root"/one_pos_whole_graph-${num_edge_samples}.txt"

min_src_out_degree=3
min_dst_in_degree=3

python -m XGCN.data.process.evaluation_set_generation \
    --file_input_graph $file_input_graph \
    --file_output_graph $file_output_graph \
    --file_output_eval_set $file_output_eval_set \
    --seed $seed --graph_type $graph_type --graph_format $graph_format \
    --num_edge_samples $num_edge_samples \
    --min_src_out_degree $min_src_out_degree \
    --min_dst_in_degree $min_dst_in_degree \
    --eval_method $eval_method \
    # --num_neg $num_neg \


##########################
file_input_graph=$file_train_graph
file_output_graph=$file_train_graph
seed=2

# available graph type: 'homo', 'user-item'
graph_type='user-item'
# available graph format: 'edge_list', 'adjacency_list'
graph_format='edge_list'

eval_method='multi_pos_whole_graph'
num_edge_samples=100000
# num_neg=1000
file_output_eval_set=$instance_root"/multi_pos_whole_graph-${num_edge_samples}.txt"

min_src_out_degree=3
min_dst_in_degree=3

python -m XGCN.data.process.evaluation_set_generation \
    --file_input_graph $file_input_graph \
    --file_output_graph $file_output_graph \
    --file_output_eval_set $file_output_eval_set \
    --seed $seed --graph_type $graph_type --graph_format $graph_format \
    --num_edge_samples $num_edge_samples \
    --min_src_out_degree $min_src_out_degree \
    --min_dst_in_degree $min_dst_in_degree \
    --eval_method $eval_method \
    # --num_neg $num_neg \
