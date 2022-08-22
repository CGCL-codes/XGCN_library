from utils import io

import argparse


def _parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--file_input", type=str)
    parser.add_argument("--file_output", type=str)
    parser.add_argument("--file_output_2", type=str)
    
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--raw_data_root", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--results_root", type=str)
    
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_sample", type=int)
    
    parser.add_argument("--num_part", type=int)
    parser.add_argument("--part_method", type=str)
    
    parser.add_argument("--validation_method", type=str)
    parser.add_argument("--test_method", type=str)
    parser.add_argument("--file_validation", type=str)
    parser.add_argument("--file_test", type=str)
    parser.add_argument("--mask_nei_when_validation", type=int)
    parser.add_argument("--mask_nei_when_test", type=int)
    parser.add_argument("--val_batch_size", type=int)
    parser.add_argument("--test_batch_size", type=int)
    
    parser.add_argument("--not_eval", type=int)
    parser.add_argument("--epochs_need_save", type=str)
    
    parser.add_argument("--model", type=str)
    parser.add_argument("--train_dl", type=str)
    parser.add_argument("--train_batch_size", type=int)
    parser.add_argument("--ensure_neg_is_not_neighbor", type=int)
    parser.add_argument("--edge_sample_ratio", type=float)
    
    parser.add_argument("--topk", type=int)
    parser.add_argument("--use_uniform_weight", type=int)
    parser.add_argument("--ppr_data_root", type=str)
    
    parser.add_argument("--num_gcn_layers", type=int)
    parser.add_argument("--stack_layers", type=int)  # bool
    
    parser.add_argument("--num_layer_sample", type=str)  # e.g. "[10, 10]"
    parser.add_argument("--gnn_arch", type=str)

    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--emb_init_std", type=float)
    parser.add_argument("--emb_lr", type=float)
    parser.add_argument("--from_pretrained", type=int)
    parser.add_argument("--file_pretrained_emb", type=str)
    parser.add_argument("--freeze_emb", type=int)
    parser.add_argument("--l2_reg_weight", type=float)
    parser.add_argument("--loss_fn", type=str)
    parser.add_argument("--margin", type=float)
    
    parser.add_argument("--use_sparse", type=int)
    
    parser.add_argument("--device", type=str)
    parser.add_argument("--emb_table_device", type=str)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--use_numba_csr_mult", type=int)

    parser.add_argument("--dnn_arch", type=str)
    parser.add_argument("--scale_net_arch", type=str)
    parser.add_argument("--use_two_dnn", type=int)
    parser.add_argument("--use_identical_dnn", type=int)
    parser.add_argument("--use_special_dnn", type=int)
    parser.add_argument("--dnn_lr", type=float)
    parser.add_argument("--dnn_l2_reg_weight", type=float)
    parser.add_argument("--use_dnn_list", type=int)
    
    parser.add_argument("--renew_by_check_val_score", type=int)
    parser.add_argument("--endure", type=int)
    parser.add_argument("--renew_by_loading_best", type=int)
    parser.add_argument("--renew_and_prop_freq", type=int)
    parser.add_argument("--max_renew_times", type=int)
    parser.add_argument("--max_prop_times", type=int)
    parser.add_argument("--prop_type", type=str)
    parser.add_argument("--prop_times", type=int)
    parser.add_argument("--cancel_prop", type=int)
    parser.add_argument("--cancel_renew", type=int)
    parser.add_argument("--use_item2item_graph_for_item_prop", type=int)
    parser.add_argument("--zero_degree_zero_emb", type=int)
    
    parser.add_argument("--K", type=int)
    
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--val_freq", type=int)
    parser.add_argument("--convergence_threshold", type=int)
    parser.add_argument("--key_score_metric", type=str)

    parser.add_argument("--p", type=float)
    parser.add_argument("--q", type=float)
    parser.add_argument("--walk_length", type=int)
    parser.add_argument("--num_walks", type=int)
    parser.add_argument("--context_size", type=int)
    
    parser.add_argument("--file_ultra_constrain_mat", type=str)
    parser.add_argument("--file_ii_topk_neighbors", type=str)
    parser.add_argument("--file_ii_topk_similarity_scores", type=str)
    
    parser.add_argument("--num_neg", type=int)
    parser.add_argument("--neg_weight", type=float)
    parser.add_argument("--w1", type=float)
    parser.add_argument("--w2", type=float)
    parser.add_argument("--w3", type=float)
    parser.add_argument("--w4", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lambda", type=float)
    
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--theta", type=float)
    parser.add_argument("--tao", type=float)
    
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
        config.update(cmd_arg)
    else:
        config = cmd_arg
    
    return config
