# set to your own path:
all_data_root="home/sxr/MING_X/XGCN_library/XGCN_data"

dataset='amazon-book'

###### process graph for training
file_input_graph=$all_data_root"/dataset/raw_${dataset}/train.txt"
data_root=$all_data_root"/dataset/instance_${dataset}"

mkdir -p $data_root  # make sure to setup the directory

graph_type="user-item"
graph_format="adjacency_list"

python -m XGCN.data.process.process_int_graph \
    --file_input_graph $file_input_graph --data_root $data_root \
    --graph_type $graph_type --graph_format $graph_format \


###### process test set
file_input=$all_data_root"/dataset/raw_${dataset}/test.txt"
file_output=$all_data_root"/dataset/instance_${dataset}/test.pkl"

eval_method="multi_pos_whole_graph"

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --eval_method $eval_method \


# The LightGCN paper also use the test set for validation during the model training. 
# The test process is time-consuming, so we sample a portion of samples from the 
# test set for validation. 
###### sample from the test set
python sample_from_test_set_for_validation.py \
    --file_input $all_data_root"/dataset/instance_${dataset}/test.pkl" \
    --file_output $all_data_root"/dataset/instance_${dataset}/val.pkl" \
    --num_sample 3000 \
