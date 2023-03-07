project_root="/media/xreco/DEV/xiran/code/XGCN"
all_data_root="/media/xreco/DEV/xiran/data/XGCN"

# Input:
#   1. ppr data
#   2. topk
#   3. results_root
# Output:
#   csr graph constructed by ppr topk nodes

dataset="facebook"

ppr_data_root=$all_data_root"/model_output/"$dataset"/PPR/undirected-top100"
topk=32
results_root=$ppr_data_root"/graph-top${topk}"

python $project_root/model/PPR/graph_reshape/to_ppr_graph.py $project_root \
    --graph_type 'homo' \
    --ppr_data_root $ppr_data_root --results_root $results_root \
    --topk $topk \
