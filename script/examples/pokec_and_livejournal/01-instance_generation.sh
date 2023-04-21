# set to your own path:
all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'

for dataset in 'pokec' 'livejournal'
do

############## process graph for training ##############

data_root=$all_data_root'/dataset/instance_'$dataset
file_input_graph=$data_root'/train.txt'

graph_type='homo'
graph_format='edge_list'

python -m XGCN.data.process.process_int_graph \
    --file_input_graph $file_input_graph --data_root $data_root \
    --graph_type $graph_type --graph_format $graph_format \

############## process validation set ##############

file_input=$data_root'/validation.txt'
file_output=$data_root'/val_edges.pkl'

eval_method='one_pos_whole_graph'

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --eval_method $eval_method \

############## process test set ##############

file_input=$data_root'/test.txt'
file_output=$data_root'/test_set.pkl'

eval_method='multi_pos_whole_graph'

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --eval_method $eval_method \

done
