PROJECT_ROOT='/home/sxr/code/xgcn'
ALL_DATA_ROOT='/home/sxr/data/social_and_user_item'

ALL_DATASETS_ROOT=$ALL_DATA_ROOT'/datasets'
ALL_RESULTS_ROOT=$ALL_DATA_ROOT'/model_outputs'

FILE_INPUT='/home/sxr/data/word_embedding/glove.6B.300d.txt'
RESULTS_ROOT='/home/sxr/data/word_embedding/glove.6B.300d'

python $PROJECT_ROOT'/'data/NLP/process_pretrained_word_emb.py $PROJECT_ROOT \
    --file_input $FILE_INPUT \
    --results_root $RESULTS_ROOT \
