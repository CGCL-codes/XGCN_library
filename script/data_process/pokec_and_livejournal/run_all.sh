all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data

bash download.sh $all_data_root

bash process_txt_graph.sh $all_data_root

bash process_eval_sets.sh $all_data_root
