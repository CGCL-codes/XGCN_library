import sys
PROJECT_ROOT = sys.argv[1]
sys.path.append(PROJECT_ROOT)

from utils import io
from utils.parse_arguments import parse_arguments
from utils.utils import print_dict, ensure_dir
from utils.utils import set_random_seed
import helper
from train.Trainer import Trainer

import os.path as osp
import setproctitle
import time


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    setproctitle.setproctitle(config['model'] + '-' +
                              time.strftime("%d%H%M%S", time.localtime(time.time())))
    
    seed = config['seed'] if 'seed' in config else 2022
    set_random_seed(seed)
    
    data_root = config['data_root']
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    data = {}
    data['info'] = io.load_yaml(osp.join(data_root, 'info.yaml'))
    
    if config['model'] == 'node2vec':
        model = helper.build_model(config, data)
        train_dl = helper.build_train_dl(config, data)
    else:
        train_dl = helper.build_train_dl(config, data)
        model = helper.build_model(config, data)
    
    opt = helper.build_optimizer(config, data)
    
    val_dl, test_dl = helper.build_val_test_dl(config, data)
    
    trainer = Trainer(
        config=config,
        data = data,
        model=model,
        opt=opt,
        train_dl=train_dl, val_dl=val_dl, test_dl=test_dl
    )
    
    trainer.train()
    
    trainer.test()


if __name__ == "__main__":
    
    main()
