all_data_root=/home/sxr/code/XGCN_and_data/XGCN_data

mkdir -p $all_data_root/dataset
cd  $all_data_root/dataset

wget https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip

unzip instance_pokec_and_livejournal.zip
