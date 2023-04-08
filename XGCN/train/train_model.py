import XGCN
from XGCN.utils.utils import print_dict, ensure_dir, set_random_seed
from XGCN.data import io

import os.path as osp


def train_model(config):
    print("##### Model Config #####")
    print_dict(config)
    
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    seed = config['seed'] if 'seed' in config else 1999
    set_random_seed(seed)
    
    data = {}  # containing some global data objects
    
    model = XGCN.create_Model(config, data)
    
    train_dl = XGCN.create_DataLoader(config, data)
    
    val_evaluator = XGCN.create_val_Evaluator(config, data, model)
    test_evaluator = XGCN.create_test_Evaluator(config, data, model)
    
    trainer = XGCN.create_Trainer(config, data, model, train_dl,
                                 val_evaluator, test_evaluator)
    
    trainer.train_and_test()
    
    return model
