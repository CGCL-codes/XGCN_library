file_input='/home/sxr/code/XGCN_and_data/XGCN_data/dataset/raw_gowalla/test.txt'
file_output='/home/sxr/code/XGCN_and_data/XGCN_data/dataset/instance_gowalla/_test.pkl'

evaluation_method='multi_pos_whole_graph'

python -m XGCN.data.process.process_evaluation_set \
    --file_input $file_input --file_output $file_output \
    --evaluation_method $evaluation_method \
