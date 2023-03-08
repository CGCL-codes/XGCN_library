all_data_root=$1

mkdir -p $all_data_root/dataset
cd  $all_data_root/dataset

wget https://data4public.blob.core.windows.net/xgcn/instance_pokec_and_livejournal.zip

unzip instance_pokec_and_livejournal.zip
