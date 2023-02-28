from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import print_dict, ensure_dir, set_random_seed
from XGCN.utils import io
from XGCN.model.build import build_Model
from XGCN.dataloading.build import build_DataLoader
from XGCN.evaluator.build import build_val_Evaluator, build_test_Evaluator
from XGCN.train.Trainer import build_Trainer

import os.path as osp


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    seed = config['seed'] if 'seed' in config else 1999
    set_random_seed(seed)
    
    data = {}  # containing some global data objects
    data['data_root'] = config['data_root']
    
    model = build_Model(config, data)
    
    train_dl = build_DataLoader(config, data)
    
    val_evaluator = build_val_Evaluator(config, data, model)
    test_evaluator = build_test_Evaluator(config, data, model)
    
    trainer = build_Trainer(config, data, model, train_dl,
                            val_evaluator, test_evaluator)
    
    trainer.train_and_test()


if __name__ == '__main__':
    
    main()
