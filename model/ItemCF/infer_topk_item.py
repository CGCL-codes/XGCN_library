import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from model.ItemCF.ItemCF import ItemCF

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from utils.utils import get_formatted_results
from helper.build_val_test_dl import build_eval_dl
from helper.eval_helper import eval_model

import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    model = ItemCF(config, {})
    
    print("# infer top-k...")
    model.infer_top_k_item_from_file_and_save(
        config['topk'], config['file_input'], config['file_output']
    )


if __name__ == "__main__":
    
    setproctitle.setproctitle('xiran-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
