all_data_root='/media/xreco/DEV/xiran/code/XGCN_data'

raw_data_root=$all_data_root'/dataset/raw_facebook'
mkdir -p $raw_data_root

cd $raw_data_root
wget 'http://snap.stanford.edu/data/facebook_combined.txt.gz'
gunzip 'facebook_combined.txt.gz'

mv 'facebook_combined.txt' 'raw_graph.txt'

echo 'First 5 lines of the .txt graph file:'
head -n 5 'raw_graph.txt'
