PROJECT_ROOT='/home/sxr/code/xgcn'

CONFIG_FILE='/home/sxr/code/xgcn/data/NLP/process_feat.yaml'
FILE_INPUT='/home/sxr/data/social_and_user_item/raw_datasets/MovieLens-20m/ml-20m.item'
DATA_ROOT='/home/sxr/data/word_embedding/glove.6B.300d'
RESULTS_ROOT='/home/sxr/data/social_and_user_item/datasets/instance_MovieLens-20m/processed_feat/item-300d'

python $PROJECT_ROOT'/'data/NLP/process_feat.py $PROJECT_ROOT \
    --config_file $CONFIG_FILE \
    --file_input $FILE_INPUT \
    --data_root $DATA_ROOT \
    --results_root $RESULTS_ROOT \
