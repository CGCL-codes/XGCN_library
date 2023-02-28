project_root="/media/xreco/DEV/xiran/code/XGCN"
all_data_root="/media/xreco/DEV/xiran/data/XGCN"

dataset="facebook"
model="Node2vec"

data_root=$all_data_root"/dataset/instance_"$dataset

seed=1999

device="cuda:0"

results_root=$all_data_root"/model_output/"$dataset"/"$model"/[seed$seed]"

python $project_root/main/main.py $project_root \
    --data_root $data_root --results_root $results_root \
    --seed $seed \
    --device $device \
    --model $model \
    --val_evaluator "WholeGraph_MultiPos_Evaluator" --val_batch_size 256 \
    --file_val_set $data_root"/val_set.pkl" \
    --test_evaluator "WholeGraph_MultiPos_Evaluator" --test_batch_size 256 \
    --file_test_set $data_root"/test_set.pkl" \
    --train_batch_size 2048 --num_workers 4 \
    --epochs 200 --val_freq 1 \
    --key_score_metric 'r100' \
    --convergence_threshold 20 \
    --emb_dim 64 --emb_lr 0.005 \
    --num_neg 5 \
    --walk_length 16 \
    --num_walks 8 \
    --context_size 5 \
    --p 1.0 --q 10.0 \
