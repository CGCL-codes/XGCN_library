project_root="/media/xreco/DEV/xiran/code/XGCN"
all_data_root="/media/xreco/DEV/xiran/data/XGCN"

all_dataset_root=$all_data_root'/dataset'
all_results_root=$all_data_root'/model_output'

DATASET='pokec'
RESULTS_DIR='UltraGCN/data'

python $project_root/XGCN/model/UltraGCN/prepare_ultragcn_data.py \
    --data_root $all_dataset_root'/instance_'$DATASET \
    --results_root $all_results_root'/gnn_'$DATASET'/'$RESULTS_DIR \
    --topk 32 \
    --graph_type 'homo' \
