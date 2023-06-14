from XGCN.data import io

import argparse


def _parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--results_root", type=str)
    
    parser.add_argument("--file_input", type=str)
    parser.add_argument("--file_input_graph", type=str)
    parser.add_argument("--file_output", type=str)
    parser.add_argument("--file_output_graph", type=str)
    parser.add_argument("--file_output_eval_set", type=str)
    parser.add_argument("--eval_method", type=str)
    parser.add_argument("--num_edge_samples", type=int)
    parser.add_argument("--graph_format", type=str)
    parser.add_argument("--graph_type", type=str)
    
    parser.add_argument("--num_sample", type=int)
    parser.add_argument("--min_src_out_degree", type=int)
    parser.add_argument("--min_dst_in_degree", type=int)
    parser.add_argument("--num_validation", type=int)
    
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--Dataset_type", type=str)
    parser.add_argument("--train_num_layer_sample", type=str)
    parser.add_argument("--num_gcn_layers", type=int)
    parser.add_argument("--NodeListDataset_type", type=str)
    parser.add_argument("--pos_sampler", type=str)
    parser.add_argument("--neg_sampler", type=str)
    parser.add_argument("--num_neg", type=int)
    parser.add_argument("--BatchSampleIndicesGenerator_type", type=str)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--epoch_sample_ratio", type=float)
    parser.add_argument("--str_num_total_samples", type=str)
    parser.add_argument("--val_method", type=str)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--file_val_set", type=str)
    parser.add_argument("--test_method", type=str)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--file_test_set", type=str)
    parser.add_argument("--epochs", type=int)
    
    parser.add_argument("--use_validation_for_early_stop", type=int)
    parser.add_argument("--val_freq", type=int)
    parser.add_argument("--key_score_metric", type=str)
    parser.add_argument("--convergence_threshold", type=int)
    parser.add_argument("--forward_mode", type=str)
    
    parser.add_argument("--device", type=str)
    parser.add_argument("--graph_device", type=str)
    parser.add_argument("--emb_table_device", type=str)
    parser.add_argument("--gnn_device", type=str)
    parser.add_argument("--out_emb_table_device", type=str)
    
    parser.add_argument("--from_pretrained", type=int)  # bool
    parser.add_argument("--file_pretrained_emb", type=str)
    parser.add_argument("--freeze_emb", type=int)  # bool
    parser.add_argument("--use_sparse", type=int)  # bool
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--emb_init_std", type=float)
    parser.add_argument("--emb_lr", type=float)
    parser.add_argument("--gnn_arch", type=str)
    parser.add_argument("--gnn_lr", type=float)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--L2_reg_weight", type=float)
    parser.add_argument("--use_ego_emb_L2_reg", type=int)
    parser.add_argument("--infer_num_layer_sample", type=str)
    
    parser.add_argument("--topk", type=int)
    parser.add_argument("--num_walks", type=int)
    parser.add_argument("--walk_length", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--margin", type=float)
    
    parser.add_argument("--stack_layers", type=int)
    
    parser.add_argument("--ppr_data_root", type=str)
    parser.add_argument("--ppr_data_device", type=str)
    parser.add_argument("--forward_device", type=str)
    parser.add_argument("--use_uniform_weight", type=int)
    
    parser.add_argument("--context_size", type=int)
    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)

    parser.add_argument("--partition_cache_filepath", type=str)
    parser.add_argument("--num_parts", type=int)
    parser.add_argument("--group_size", type=int)
    parser.add_argument("--subgraph_device", type=str)
    
    parser.add_argument("--file_ultra_constrain_mat", type=str)
    parser.add_argument("--file_ii_topk_neighbors", type=str)
    parser.add_argument("--file_ii_topk_similarity_scores", type=str)
    
    parser.add_argument("--neg_weight", type=float)
    parser.add_argument("--lambda", type=float)
    parser.add_argument("--gamma", type=float)

    parser.add_argument("--hidden", type=int)
    parser.add_argument("--n_layers_1", type=int)
    parser.add_argument("--n_layers_2", type=int)
    parser.add_argument("--pre_process", type=int)
    parser.add_argument("--residual", type=int)
    parser.add_argument("--bns", type=int)
    
    parser.add_argument("--num_dnn_layers", type=int)
    
    parser.add_argument("--rmax_ratio", type=float)
    
    parser.add_argument("--dnn_lr", type=float)
    parser.add_argument("--dnn_arch", type=str)
    parser.add_argument("--use_scale_net", type=int)
    parser.add_argument("--scale_net_arch", type=str)
    parser.add_argument("--renew_by_loading_best", type=int)
    parser.add_argument("--K", type=int)
    parser.add_argument("--T", type=int)
    parser.add_argument("--tolerance", type=int)
    
    parser.add_argument("--only_use_heuristic", type=int)
    parser.add_argument("--heuristic_type", type=str)
    parser.add_argument("--max_hash_hops", type=int)
    parser.add_argument("--minhash_num_perm", type=int)
    parser.add_argument("--hll_p", type=int)
    parser.add_argument("--use_zero_one", type=int)
    parser.add_argument("--p_drop", type=float)

    (args, unknown) = parser.parse_known_args()
    parsed_results = {}
    for arg in sorted(vars(args)):
        value = getattr(args, arg)
        if value is not None:
            parsed_results[arg] = '' if value  in ['none', 'None'] else value
    
    return parsed_results


def parse_arguments():

    cmd_arg = _parse_arguments()
    
    if 'config_file' in cmd_arg:
        config = io.load_yaml(cmd_arg['config_file'])
        if config is None:
            config = cmd_arg
        else:
            config.update(cmd_arg)
    else:
        config = cmd_arg
    
    return config
