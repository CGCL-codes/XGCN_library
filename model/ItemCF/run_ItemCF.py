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
    
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    data = {}
    
    model = ItemCF(config, data)
    data['model'] = model
    
    eval_dl = build_eval_dl(
        eval_method=config['test_method'],
        file_eval=config['file_test'],
        mask_nei=True,
        batch_size=256,
        data=data
    )
    
    results = eval_model(model, eval_dl)
    
    results['formatted'] = get_formatted_results(results)
    print(results)
    io.save_json(osp.join(results_root, "test_results.json"), results)


if __name__ == "__main__":
    
    setproctitle.setproctitle('xiran-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
