all_data_root='/home/sxr/code/XGCN_and_data/XGCN_data'
config_file_root='/home/sxr/code/XGCN_and_data/XGCN_library/config'

dataset=facebook
model=NeighborBased
use_jaccard=0

data_root=$all_data_root/dataset/instance_$dataset
results_root=$all_data_root/model_output/$dataset/$model/[use_jaccard${use_jaccard}]

python -m XGCN.model.NeighborBased.run_NeighborBased \
    --data_root $data_root --results_root $results_root \
    --config_file $config_file_root/$model-config.yaml \
    --test_method 'multi_pos_whole_graph' \
    --file_test_set $data_root/test-multi_pos_whole_graph.pkl \
    --use_jaccard $use_jaccard \
