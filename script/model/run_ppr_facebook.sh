project_root="/media/xreco/DEV/xiran/code/gnn_zoo"
all_data_root="/media/xreco/DEV/xiran/data/gnn_zoo"

dataset="facebook"

data_root=$all_data_root"/dataset/instance_"$dataset

results_root=$all_data_root"/model_output/"$dataset"/PPR/undirected-top100"

python $project_root/model/PPR/run_ppr.py $project_root \
    --data_root $data_root --results_root $results_root \
    --topk 100 \
    --num_walks 1000 \
    --walk_length 30 \
    --alpha 0.3 \
