###### process graph for training
# set to your own path:
file_input_graph='/home/xxx/XGCN_data/dataset/raw_facebook/train.txt'
data_root='/home/xxx/XGCN_data/dataset/instance_facebook'

mkdir -p $data_root  # make sure to setup the directory

graph_type='homo'
graph_format='edge_list'

python -m XGCN.data.process.process_int_graph \
    --file_input_graph $file_input_graph --data_root $data_root \
    --graph_type $graph_type --graph_format $graph_format \

###### process validation set
file_input='/home/xxx/XGCN_data/dataset/raw_facebook/val-one_pos_k_neg.txt'
file_output='/home/xxx/XGCN_data/dataset/instance_facebook/val-one_pos_k_neg.pkl'

evaluation_method='one_pos_k_neg'

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --evaluation_method $evaluation_method \

###### process test set
file_input='/home/xxx/XGCN_data/dataset/raw_facebook/test-multi_pos_whole_graph.txt'
file_output='/home/xxx/XGCN_data/dataset/instance_facebook/test-multi_pos_whole_graph.pkl'

evaluation_method='multi_pos_whole_graph'

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --evaluation_method $evaluation_method \
