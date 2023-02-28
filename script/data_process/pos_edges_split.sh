project_root='/media/xreco/DEV/xiran/code/gnn_zoo'
all_data_root='/media/xreco/DEV/xiran/data/gnn_zoo'

# Input:
#   $data_root: contains the graph to be split: (info.yaml, indptr.pkl, indices.pkl)
# Output:
#   Under the directory $results_root, these files will be saved:
#       1. pos_edges.pkl: the removed edges
#       2. (info.yaml, indptr.pkl, indices.pkl): the new graph

dataset='facebook'
raw_data_root=$all_data_root'/dataset/raw_'$dataset'/'

data_root=$raw_data_root'/csr'
results_root=$all_data_root'/dataset/instance_'$dataset

python $project_root/data_process/pos_edges_split/pos_edges_split.py $project_root \
    --data_root $data_root \
    --results_root $results_root \
    --seed 1999 \
    --num_sample 5000 \
    --min_src_out_degree 3 \
    --min_dst_in_degree 1 \
