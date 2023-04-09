from XGCN.data import io
from XGCN.utils.utils import get_formatted_results

import os.path as osp


class TrainTracer:
    
    def __init__(self, data, model,
                 key_score_metric, convergence_threshold, results_root):
        assert hasattr(model, 'save')  # save the model for future inference
        
        self.data = data
        self.model = model
        
        self.key_score_metric = key_score_metric
        self.convergence_threshold =convergence_threshold
        self.record_root = results_root
        
        self.record_file = osp.join(self.record_root, "train_record.txt")
        self.best_record_file = osp.join(self.record_root, "train_record_best.json")
        
        self.best_key_score = None
        self.epoch_best = None
        
        self.data['epoch_best'] = 0
        
    def check_and_save(self, epoch, val_results: dict):
        val_results['epoch'] = epoch
        
        self._write_to_record_file(val_results)
        
        self._check_score_and_save(epoch, val_results)
        
        is_converged = self._check_if_converged(epoch)
        
        return is_converged

    def _write_to_record_file(self, val_results):
        if self.epoch_best is None:
            with open(self.record_file, "w") as f:
                f.write(','.join(val_results.keys()) + '\n')  # first line: metric names
        
        with open(self.record_file, "a") as f:
            f.write(
                ','.join(map(
                    lambda x: "{:.4g}".format(x) if isinstance(x, float) else str(x), 
                    val_results.values())
                ) + '\n'
            )

    def _check_score_and_save(self, epoch, val_results):
        key_score = val_results[self.key_score_metric]
        
        if self.best_key_score is None or key_score > self.best_key_score:
            print(">> new best score -", self.key_score_metric, ":", key_score)
            self.best_key_score = key_score
            self.epoch_best = epoch
            
            val_results['formatted'] = get_formatted_results(val_results)
            # save the best validation score up till now
            io.save_json(self.best_record_file, val_results)
            
            # save the best model up till now
            self.model.save()
        else:
            print(">> distance_between_best_epoch:", epoch - self.epoch_best, "threshold:", self.convergence_threshold)
        
        self.data['epoch_best'] = self.epoch_best

    def _check_if_converged(self, epoch):
        is_converged = False
        if epoch - self.epoch_best >= self.convergence_threshold:
            print("converged at epoch", epoch)
            is_converged = True
        return is_converged
