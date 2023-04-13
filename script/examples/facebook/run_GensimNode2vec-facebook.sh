project_root="/home/sxr/code/XGCN"
all_data_root="/home/sxr/data/XGCN"

dataset="facebook"
model="GensimNode2vec"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

results_root=$all_data_root"/model_output/"$dataset"/"$model"/[seed$seed]"

python $project_root/main/main.py $project_root \
    --config_file $project_root"/model/"$model"/config.yaml" \
    --seed $seed \
    --data_root $data_root --results_root $results_root \
    --val_method "one_pos_k_neg" --val_batch_size 256 \
    --file_val_set $data_root"/val-one_pos_k_neg.pkl" \
    --test_method "multi_pos_whole_graph" --test_batch_size 256 \
    --file_test_set $data_root"/test-multi_pos_whole_graph.pkl" \
    --epochs 200 --val_freq 1 \
    --emb_dim 64 --emb_lr 0.01 \
    --num_walks 16 --walk_length 16 \
    --p 1.0 --q 1.0 \
    --context_size 5 --num_neg 5 \
    --num_workers 6 \
