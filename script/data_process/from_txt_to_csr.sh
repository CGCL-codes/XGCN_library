project_root='/media/xreco/DEV/xiran/code/gnn_zoo'
all_data_root='/media/xreco/DEV/xiran/data/gnn_zoo'

# Input: 
#   a .txt file (edges or adjacency list)
# Output: 
#   3 files under the directory $results_root: (info.yaml, indptr.pkl, indices.pkl)

dataset='facebook'
file_txt_graph='facebook_combined.txt'

raw_data_root=$all_data_root'/dataset/raw_'$dataset'/'

file_input=$raw_data_root'/'$file_txt_graph
results_root=$raw_data_root'/csr'

python $project_root/data_process/from_txt_to_csr/from_txt_to_csr.py $project_root \
    --file_input $file_input \
    --results_root $results_root \
    --is_adj_list 0 \
    --graph_type 'homo' \
