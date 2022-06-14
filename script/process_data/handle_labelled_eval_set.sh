PROJECT_ROOT='/media/xreco/jianxun/xGCN'
# ALL_DATA_ROOT='/media/xreco/jianxun/xgcn_data'
ALL_DATA_ROOT='/media/xreco/DEV/xiran/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'

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

DATASET='xbox-100m'
DATA_ROOT=$ALL_DATASETS_ROOT'/instance_'$DATASET

python ../data/handle_labelled_eval_set.py $PROJECT_ROOT \
    --file_input '/media/xreco/DEV/socgraph/production_full/social_graph_v2/test' \
    --file_output $DATA_ROOT'/test-1-99.pkl' \
    --file_output_2 $DATA_ROOT'/test-1-99-pos_edges.pkl' \

###### for model training:
#   --validation_method 'one_pos_k_neg' \
#   --mask_nei_when_validation 0 \
#   --file_validation $DATA_ROOT'/test-1-99.pkl' \
#   --test_method 'one_pos_k_neg' \
#   --mask_nei_when_test 0 \
#   --file_test $DATA_ROOT'/test-1-99.pkl' \
######
