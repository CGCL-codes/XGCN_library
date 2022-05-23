from utils import io

import os.path as osp
from copy import deepcopy


class TrainTracer:
    
    def __init__(self, config, data,
                 record_root, fn_save_best_model=None):
        self.config = config
        self.data = data
        
        self.fn_save_best_model = fn_save_best_model
        self.convergence_threshold = config['convergence_threshold']
        self.record_file = osp.join(record_root, "train_record.txt")
        self.best_record_file = osp.join(record_root, "train_record_best.json")
        
        self.best_key_score = None
        self.epoch_best = None
        
        self.data['epoch_best'] = 0
        
    def check_and_save(self, key_score, epoch, val_results: dict, fn_save_best_model=None):
        _results = deepcopy(val_results)
        _results['epoch'] = epoch
        
        if self.epoch_best is None:
            with open(self.record_file, "w") as f:
                f.write(','.join(_results.keys()) + '\n')
                
        with open(self.record_file, "a") as f:
            f.write(
                ','.join(map(
                    lambda x: "{:.4g}".format(x) if isinstance(x, float) else str(x), 
                    _results.values())
                ) + '\n'
            )
        
        if self.best_key_score is None or key_score > self.best_key_score:
            print(">> new best score:", key_score)
            self.best_key_score = key_score
            self.epoch_best = epoch
            
            _results = deepcopy(val_results)
            _results['epoch_best'] = self.epoch_best
            io.save_json(self.best_record_file, _results)
            
            if fn_save_best_model is not None:
                fn_save_best_model()
            else:
                self.fn_save_best_model()
        else:
            print(">> distance_between_best_epoch:", epoch - self.epoch_best, "threshold:", self.convergence_threshold)
        
        self.data['epoch_best'] = self.epoch_best
        
        is_converged = False
        if epoch - self.epoch_best >= self.convergence_threshold:
            print("converged at epoch {}".format(epoch))
            is_converged = True

        return is_converged
