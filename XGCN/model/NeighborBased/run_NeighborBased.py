import XGCN
from XGCN.model.NeighborBased import NeighborBased
from XGCN.data import io
from XGCN.utils.parse_arguments import parse_arguments
from XGCN.utils.utils import print_dict, ensure_dir, get_formatted_results

import os.path as osp


def main():
    
    config = parse_arguments()
    print_dict(config)
    
    results_root = config['results_root']
    ensure_dir(results_root)
    io.save_yaml(osp.join(results_root, 'config.yaml'), config)
    
    model = NeighborBased(config)
    
    test_evaluator = XGCN.create_test_Evaluator(
        config=config, data={}, model=model
    )
    results = test_evaluator.eval(desc='test')
    
    results['formatted'] = get_formatted_results(results)
    print(results)
    io.save_json(osp.join(results_root, "test_results.json"), results)


if __name__ == "__main__":
    
    main()
