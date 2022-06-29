from .TrainTracer import TrainTracer
from utils import io
from utils.Timer import Timer
from utils.utils import get_formatted_results
from helper.eval_helper import eval_model

import numpy as np
import torch
import os.path as osp
from tqdm import tqdm


class Trainer:
    
    def __init__(self, config, data,
                 model,
                 opt, 
                 train_dl, val_dl, test_dl,
                 ):
        '''
        requirements:
        
        config:
            results_root: 
            convergence_threshold: 
            val_freq: 
            key_score_metric: 
            epochs: 
            
        '''
        self.config = config
        self.data = data
        
        self.model = model
        self.opt = opt
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.results_root = config['results_root']
        
        def save_best_model():
            self.model.save(root=self.results_root)
            
        def load_best_model():
            self.model.load(root=self.results_root)
        
        self.train_tracer = TrainTracer(config,
                                        data,
                                        fn_save_best_model=save_best_model,
                                        record_root=self.results_root)
        self.load_best_model = load_best_model
        
        self.timer = Timer(record_root=self.results_root)
        
        self.key_score_metric = config['key_score_metric']
        self.val_freq = config['val_freq']
        self.epochs = config['epochs']
    
    def _train_loop(self):
        for epoch in range(self.epochs):
            
            self.data['epoch'] = epoch
            
            if 'not_eval' in self.config and self.config['not_eval']:
                if epoch in eval(self.config['epochs_need_save']):
                    print("## save model at epoch", epoch)
                    self.model.prepare_for_eval()
                    self.model.save(self.config['results_root'],
                                    'out_emb_table-epoch' + str(epoch) + '.pt')
            else:
                if epoch % self.val_freq == 0:
                    self.timer.start("val")
                    with torch.no_grad():
                        print("epoch {} val...".format(epoch))
                        self.model.prepare_for_eval()
                        
                        results = eval_model(self.model, self.val_dl, desc='val')
                        
                        key_score = results[self.key_score_metric]
                        print("val:", results)
                        results.update({"loss": np.nan if epoch == 0 else epoch_loss})
                        
                        is_converged = self.train_tracer.check_and_save(key_score, epoch, results)
                        if is_converged:
                            break
                    self.timer.end("val")

            epoch_loss = []
            self.model.prepare_for_train()
            
            print(">> epoch {}".format(epoch + 1))
            self.timer.start("epoch")
            for batch_data in tqdm(self.train_dl):
                self.timer.start("batch")
                
                self.timer.start("batch_forward")
                loss = self.model(batch_data)
                self.timer.end("batch_forward")
                
                self.timer.start("batch_backward")
                self.opt.zero_grad()
                loss.backward()
                
                if hasattr(self.opt, 'clip_grad_norm'):
                    self.opt.clip_grad_norm()
                
                self.opt.step()
                epoch_loss.append(loss.item())
                self.timer.end("batch_backward")
                
                self.timer.end("batch")
                self.timer.save_record()
            self.timer.end("epoch")
            self.timer.save_record()
            
            epoch_loss = np.mean(epoch_loss)
            print("loss {}".format(epoch_loss))
        
    def train(self):
        self.timer.start("train")
        
        try:
            self._train_loop()
        except KeyboardInterrupt:
            pass
        
        self.timer.end("train")
        self.timer.save_record(root=self.results_root)
        
    def test(self):
        if 'not_eval' in self.config and self.config['not_eval']:
            return
        else:
            print("test...")
            self.timer.start("test")
            with torch.no_grad():
                self.load_best_model()
                
                results = eval_model(self.model, self.val_dl, desc='test')
                
                results['formatted'] = get_formatted_results(results)
                print("test:", results)
                io.save_json(osp.join(self.results_root, "test_results.json"), results)
            self.timer.end("test")
            self.timer.save_record()
