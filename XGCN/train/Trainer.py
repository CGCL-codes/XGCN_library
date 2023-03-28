from .TrainTracer import TrainTracer
from XGCN.data import io
from XGCN.utils.Timer import Timer
from XGCN.utils.utils import get_formatted_results

import numpy as np
import os.path as osp
from tqdm import tqdm


def build_Trainer(config, data, model, train_dl,
                  val_evaluator, test_evaluator):
    trainer = Trainer(
        data, model, train_dl,
        epochs=config['epochs'],
        results_root=config['results_root'],
        val_evaluator=val_evaluator,
        test_evaluator=test_evaluator,
        val_freq=config['val_freq'],
        key_score_metric=config['key_score_metric'],
        convergence_threshold=config['convergence_threshold']
    )
    return trainer


class Trainer:
    
    def __init__(self, data, model, train_dl,
                 epochs, results_root,
                 test_evaluator=None,
                 val_evaluator=None,
                 val_freq=None, key_score_metric=None, convergence_threshold=None):
        self.data = data
        self.model = model
        self.train_dl = train_dl
        self.epochs = epochs
        self.results_root = results_root
        self.val_evaluator = val_evaluator
        self.test_evaluator = test_evaluator
        
        self.do_val = self.val_evaluator is not None
        
        if self.do_val:
            self.val_freq = val_freq
            self.train_tracer = TrainTracer(
                data, model,
                key_score_metric=key_score_metric,
                convergence_threshold=convergence_threshold,
                results_root=results_root
            )
        
        self.timer = Timer(record_root=self.results_root)
    
    def train(self):
        self.timer.start("train")
        if hasattr(self.model, 'on_train_begin'):
            self.model.on_train_begin()
        
        try:
            self._train_loop()
        except KeyboardInterrupt:
            pass
        
        if not self.do_val:
            self.model.save()
        
        if hasattr(self.model, 'on_train_end'):
            self.model.on_train_end()
            
        self.timer.end("train")
        self.timer.save_record()
    
    def _train_loop(self):
        for epoch in range(self.epochs):
            self.data['epoch'] = epoch
            
            if self.do_val and (epoch % self.val_freq == 0):
                self.timer.start("val")
                if hasattr(self.model, 'on_eval_begin'):
                    self.model.on_eval_begin()
                
                results = self.val_evaluator.eval(desc='val')
                
                if hasattr(self.model, 'on_eval_end'):
                    self.model.on_eval_end()
                self.timer.end("val")
                
                print("val:", results)
                results.update({"loss": np.nan if epoch == 0 else epoch_loss})
                
                is_converged = self.train_tracer.check_and_save(epoch, results)
                if is_converged:
                    break
            
            self.timer.start("epoch")
            if hasattr(self.model, 'on_epoch_begin'):
                self.model.on_epoch_begin()
            
            if hasattr(self.train_dl, 'subgraph_dl'):
                print("###----")
                with self.train_dl.subgraph_dl.enable_cpu_affinity():
                    epoch_loss = self._train_an_epoch()
            else:
                epoch_loss = self._train_an_epoch()
            
            if hasattr(self.model, 'on_epoch_end'):
                self.model.on_epoch_end()
            self.timer.end("epoch")
        
    def _train_an_epoch(self):
        print('train epoch {0}'.format(self.data['epoch'] + 1))
        if hasattr(self.model, 'train_an_epoch'):
            epoch_loss = self.model.train_an_epoch()
        else:
            loss_list = []
            for batch_data in tqdm(self.train_dl):
                self.timer.start("batch")
                
                loss = self.model.forward_and_backward(batch_data)
                loss_list.append(loss)
                
                self.timer.end("batch")
                self.timer.save_record()
            epoch_loss = np.array(loss_list).mean()
        return epoch_loss
        
    def test(self):
        self.model.load()  # load the best model on validation set
        self.timer.start("test")
        if hasattr(self.model, 'on_eval_begin'):
            self.model.on_eval_begin()

        results = self.test_evaluator.eval(desc='test')
                
        if hasattr(self.model, 'on_eval_end'):
            self.model.on_eval_end()
        self.timer.end("test")
        self.timer.save_record()
        
        results['formatted'] = get_formatted_results(results)
        print("test:", results)
        io.save_json(osp.join(self.results_root, "test_results.json"), results)

    def train_and_test(self):
        self.train()
        self.test()
