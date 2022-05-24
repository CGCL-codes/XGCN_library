PROJECT_ROOT='/media/xreco/jianxun/xgcn'
ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

DATASET='xbox-100m'

DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

# ##################################
# #### process adj graph data
# # FILE_INPUT=$DATA_ROOT'/train.txt'

# 100m graph:
FILE_INPUT='/media/xreco/DEV/socgraph/production_full/social_graph_v2/graph.tsv'

# 1m graph:
# FILE_INPUT='/media/xreco/DEV/socgraph/usconsole_sample1m/social_graph_v2/graph.tsv'

python $PROJECT_ROOT'/'data/handle_adj_graph_txt.py $PROJECT_ROOT \
    --data_root $DATA_ROOT \
    --dataset_type 'social' \
    --dataset_name $DATASET \
    --file_input $FILE_INPUT \

##################################
#### handle the old 1-99 eval set
# input format: src \t dst \t label
# each src has 1 pos and k neg
# [[src01, dst01, 1], 
#  [src01, dst02, 0], 
#   ..., 
#  [src01, dst0k, 0],
#  [src02, dst01, 1], 
#  [src02, dst02, 0], 
#   ..., 
#  [src02, dst0k, 0], ...]
# output: numpy array, [[src01, pos, neg1, ..., negk], [src02, ... ], ... ]

# 100m graph:
FILE_INPUT='/media/xreco/DEV/socgraph/production_full/social_graph_v2/test'

# # 1m graph:
# FILE_INPUT='/media/xreco/DEV/socgraph/usconsole_sample1m/social_graph_v2/test'

FILE_OUTPUT=$DATA_ROOT'/test-src_pos_neg.pkl'
FILE_OUTPUT_2=$DATA_ROOT'/test-pos_edges.pkl'

mkdir -p $DATA_ROOT

python $PROJECT_ROOT'/'data/handle_labelled_eval_set.py $PROJECT_ROOT \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \
    --file_output_2 $FILE_OUTPUT_2 \

# ##################################
# #### process eval set
# input format: src dst, each line is a positive edge
# FILE_INPUT=''
# FILE_OUTPUT=$DATA_ROOT'/test-pos_edges.pkl'

# python $PROJECT_ROOT'/'data/handle_src_pos_neg_eval_set.py $PROJECT_ROOT \
#     --file_input $FILE_INPUT \
#     --file_output $FILE_OUTPUT \

##################################
#### sample eval set
# input format: src dst, each line is a positive edge
FILE_INPUT=$DATA_ROOT'/test-pos_edges.pkl'
FILE_OUTPUT=$DATA_ROOT'/test-pos_edges-1000.pkl'
SEED=2022

python $PROJECT_ROOT'/'data/sample_numpy_array.py $PROJECT_ROOT \
    --file_input $FILE_INPUT \
    --file_output $FILE_OUTPUT \
    --seed $SEED \
    --num_sample 1000 \
