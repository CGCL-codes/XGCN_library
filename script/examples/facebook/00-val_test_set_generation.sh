# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'

############## generate validation set ##############

file_input_graph=$all_data_root'/dataset/raw_facebook/facebook_combined.txt'
graph_type='homo'
graph_format='edge_list'

seed=1999               # random seed
num_edge_samples=500    # number of edges to split
min_src_out_degree=3    # guarantee the minimum out-degree of a source node after the split
min_dst_in_degree=3     # guarantee the minimum in-degree of a destination node after the split

# available evaluation_method: 'one_pos_k_neg', 'one_pos_whole_graph', 'multi_pos_whole_graph'
eval_method='one_pos_k_neg'
num_neg=300  # the num_neg argument is required when the eval_method='one_pos_k_neg'

# the output graph will be saved as a text file in edge list format
# set to your own path:
file_output_graph=$all_data_root'/dataset/raw_facebook/train.txt'
file_output_eval_set=$all_data_root'/dataset/raw_facebook/val-one_pos_k_neg.txt'

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

############## generate test set ##############

file_input_graph=$all_data_root'/dataset/raw_facebook/train.txt'
graph_type='homo'
graph_format='edge_list'

seed=2000               # random seed
num_edge_samples=8000   # number of edges to split
min_src_out_degree=3    # guarantee the minimum out-degree of a source node after the split
min_dst_in_degree=3     # guarantee the minimum in-degree of a destination node after the split

# available evaluation_method: 'one_pos_k_neg', 'one_pos_whole_graph', 'multi_pos_whole_graph'
eval_method='multi_pos_whole_graph'
# num_neg=300  # the num_neg argument is required when the eval_method='one_pos_k_neg'

# the output graph will be saved as a text file in edge list format
# set to your own path:
file_output_graph=$all_data_root'/dataset/raw_facebook/train.txt'
file_output_eval_set=$all_data_root'/dataset/raw_facebook/test-multi_pos_whole_graph.txt'

python -m XGCN.data.process.evaluation_set_generation \
    --file_input_graph $file_input_graph \
    --file_output_graph $file_output_graph \
    --file_output_eval_set $file_output_eval_set \
    --seed $seed --graph_type $graph_type --graph_format $graph_format \
    --num_edge_samples $num_edge_samples \
    --min_src_out_degree $min_src_out_degree \
    --min_dst_in_degree $min_dst_in_degree \
    --eval_method $eval_method \
    #    --num_neg $num_neg \
