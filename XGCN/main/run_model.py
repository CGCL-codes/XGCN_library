import XGCN
from XGCN.data import io
from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import set_random_seed

import os.path as osp


def main():
    
    config = parse_arguments()

    set_random_seed(seed=config['seed'])

    model = XGCN.create_model(config)
    
    model.fit()
    
    test_results = model.test()
    print("test:", test_results)
    io.save_json(osp.join(config['results_root'], 'test_results.json'), test_results)


if __name__ == '__main__':
    
    main()
