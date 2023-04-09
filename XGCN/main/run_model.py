import XGCN
from XGCN.data import io
from XGCN.utils.parse_arguments import parse_arguments

import os.path as osp


def main():
    
    config = parse_arguments()

    model = XGCN.create_model(config)
    
    model.fit()
    
    test_results = model.test(config)
    print("test:", test_results)
    io.save_json(osp.join(config['results_root'], 'test_results.json'), test_results)


if __name__ == '__main__':
    
    main()
