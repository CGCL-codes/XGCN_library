project_root='/media/xreco/DEV/xiran/code/XGCN'
all_data_root='/media/xreco/DEV/xiran/data/XGCN'

# Input:
#   $data_root: containing pos_edges.pkl
# Output:
#   Under $data_root: val_set.pkl and test_set.pkl

dataset='facebook'
data_root=$all_data_root'/dataset/instance_'$dataset

python $project_root/data_process/from_edges_to_adj_eval_set/from_edges_to_adj_eval_set.py $project_root \
    --data_root $data_root \
    --num_validation 1000 \
