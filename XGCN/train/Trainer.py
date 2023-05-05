from .TrainTracer import TrainTracer
import XGCN
from XGCN.data import io
from XGCN.utils.Timer import Timer
from XGCN.utils.utils import get_formatted_results

import numpy as np
import os.path as osp
from tqdm import tqdm


def create_Trainer(config, data, model, train_dl):
    trainer = Trainer(
        config, data, model, train_dl,
    )
    return trainer


class Trainer:
    
    def __init__(self, config, data, model, train_dl):
        self.config = config
        self.data = data
        self.model = model
        self.train_dl = train_dl
        
        self.epochs = self.config['epochs']
        self.results_root = self.config['results_root']
        
        self.do_val = self.config['use_validation_for_early_stop']
        if self.do_val:
            self.val_method = XGCN.create_val_Evaluator(self.config, self.data, self.model)
            self.val_freq = self.config['val_freq']
            self.train_tracer = TrainTracer(
                data, model,
                key_score_metric=self.config['key_score_metric'],
                convergence_threshold=self.config['convergence_threshold'],
                results_root=self.results_root
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
            self.model.infer_out_emb_table()
            self.model.on_train_end()
            
        self.timer.end("train")
        self.timer.save_record()
    
    def _train_loop(self):
        for epoch in range(self.epochs):
            self.data['epoch'] = epoch
            
            if self.do_val and (epoch % self.val_freq == 0):
                self.timer.start("val")
                if hasattr(self.model, 'on_val_begin'):
                    self.model.on_val_begin()
                
                results = self.val_method.eval(desc='val')
                
                if hasattr(self.model, 'on_val_end'):
                    self.model.on_val_end()
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
        print('epoch {0}'.format(self.data['epoch'] + 1))
        if hasattr(self.model, 'train_an_epoch'):
            epoch_loss = self.model.train_an_epoch()
        else:
            loss_list = []
            for batch_data in tqdm(self.train_dl, desc='train'):
                self.timer.start("batch")
                
                loss = self.model.forward_and_backward(batch_data)
                loss_list.append(loss)
                
                self.timer.end("batch")
                self.timer.save_record()
            epoch_loss = np.array(loss_list).mean()
        return epoch_loss
