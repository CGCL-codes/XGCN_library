import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict
from utils.utils import combine_dict_list_and_calc_mean, get_formatted_results
from helper.build_val_test_dl import build_eval_dl
from model.BaseEmbeddingModel import BaseEmbeddingModel

import torch
from tqdm import tqdm
import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    assert 'data_root' in config
    results_root = config['results_root']
    file_results = config['file_output']  # .json file
    
    data = {}
    
    model = BaseEmbeddingModel(config, data)
    
    data['model'] = model
    
    test_dl = build_eval_dl(
        eval_method=config['test_method'],
        file_eval=config['file_test'],
        mask_nei=config['mask_nei_when_test'],
        data=data
    )
    
    with torch.no_grad():
        model.load(results_root)
        
        batch_results_list = []
        batch_results_weights = []
        num_samples = test_dl.num_samples()
        for batch_data in tqdm(test_dl, desc="eval"):
            batch_results, num_batch_samples = model.eval_a_batch(batch_data)
            batch_results_list.append(batch_results)
            batch_results_weights.append(num_batch_samples / num_samples)
        
        results = combine_dict_list_and_calc_mean(batch_results_list, batch_results_weights)
        results['formatted'] = get_formatted_results(results)
        print("eval:", results)
        io.save_json(osp.join(results_root, file_results), results)


if __name__ == "__main__":
    
    setproctitle.setproctitle('eval-' + 
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    main()
