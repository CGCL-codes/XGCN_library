###### process graph for training
# set to your own path:
file_input_graph='/home/xxx/XGCN_data/dataset/raw_gowalla/train.txt'
data_root='/home/xxx/XGCN_data/dataset/instance_gowalla'

mkdir -p $data_root  # make sure to setup the directory

graph_type='homo'
graph_format='edge_list'

python -m XGCN.data.process.process_int_graph \
    --file_input_graph $file_input_graph --data_root $data_root \
    --graph_type $graph_type --graph_format $graph_format \


###### process test set
file_input='/home/xxx/XGCN_data/dataset/raw_gowalla/test.txt'
file_output='/home/xxx/XGCN_data/dataset/instance_gowalla/test.pkl'

eval_method='multi_pos_whole_graph'

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --eval_method $eval_method \
