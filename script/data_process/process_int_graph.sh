# input text file:
file_input='/home/sxr/code/XGCN_and_data/XGCN_data/dataset/raw_facebook/facebook_combined.txt'
# data instance root:
data_root='/home/sxr/code/XGCN_and_data/XGCN_data/dataset/instance_facebook'

# available graph type: 'homo', 'user-item'
graph_type='homo'
# available graph format: 'edge_list', 'adjacency_list'
graph_format='edge_list'

python -m XGCN.data.process.process_int_graph \
    --file_input $file_input --data_root $data_root \
    --graph_type $graph_type --graph_format $graph_format \
