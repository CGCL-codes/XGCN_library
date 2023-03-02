Model
=========================

In this part, let's dive into the implementations of a model 
by customizing a new one. 
We'll first introduce the interface functions of a model, and then 
give some implementation examples of them. 


1. Interface Overview
-----------------------------

.. image:: ../asset/overview.jpg
  :width: 600
  :alt: key modules of XGCN

As shown in the figure above, to communicate with other modules, 
plus ``__init__()``, a ``Model`` must implement five interface functions: 

(1) ``__init__()``. This function is called by ``XGCN.build_Model`` 
and is responsible for initializing model parameters and optimizers. 

(2) ``forward_and_backward()``. This function is called by ``Trainer`` 
during the batch training and is supposed to execute forward calculation and 
backward propagation. 

(3) ``eval()``. This function is called by ``Evaluator`` and is expected to return 
the corresponding prediction to calculate accuracy metrics. 

(4) ``save()``. This function is call by ``Trainer`` when a new best score on the 
validation set is achieved. The function should save the parameters that is needed 
by ``eval()``. 

(5) ``load()``. This function is call by ``Trainer`` when the training process is converged 
and the testing is to begin. The function should load the saved best parameters. 


Specifically, these functions are described by the ``BaseModel`` class 
which must be inherited by a new model. 
The code of the ``BaseModel`` class is as follows: 

.. code:: python

    class BaseModel:
        
        def __init__(self, config, data):
            # init model parameters
            # init optimizer
            pass
        
        def forward_and_backward(self, batch_data):
            # This function is called by Trainer during the training loop. 
            # Given a batch of training data,
            # perform forward process to calculate loss, 
            # and then perform backward process to update model parameters.
            # The form of batch_data depends on the configuration of DataLoader. 

            # Return loss value.
            loss = 0.0
            return loss
        
        def eval(self, batch_data):
            # This function will be called by the Evaluator.
            # The form of batch_data depends on the configuration of Evaluator, 
            # and the return value should also correspond to the Evaluator.
            output = None
            return output
        
        def save(self, root=None):
            # This function is called by Trainer to save the best model parameters during training. 
            pass
        
        def load(self, root=None):
            # After the training process converges, this function is called by Trainer 
            # to load the saved best model for testing.
            pass

XGCN provides a ``BaseEmbeddingModel`` class which is inherited from ``BaseModel`` 
and implements some useful functions for model evaluation 
(see ``XGCN/model/base/BaseEmbeddingModel.py``). 
It's easier to start from the ``BaseEmbeddingModel`` class. 
With it we only need to implement these three functions: 
``__init__()``, ``forward_and_backward()``, and ``on_eval_begin()``. 

In the following, we'll create a new model based on the ``BaseEmbeddingModel`` class, 
and implement the needed functions. 
Firstly, create a file named ``NewModel.py`` in the ``XGCN/model`` directory 
with the contents below: 

.. code:: python

    from XGCN.model.base import BaseEmbeddingModel
    from XGCN.model.module import init_emb_table, dot_product, bpr_loss
    from XGCN.utils import io

    import torch
    import os.path as osp

    class NewModel(BaseEmbeddingModel):
        
        def __init__(self, config, data):
            super().__init__(config, data)
            pass
        
        def forward_and_backward(self, batch_data):
            loss = 0.0
            return loss
        
        @torch.no_grad()
        def on_eval_begin(self):
            pass


2. Implement __init__()
-----------------------------

The ``__init__()`` function is responsible for initializing model parameters and optimizers. 
For simplicity, here we just create an embedding table, an MLP, and a Adam optimizer: 

.. code:: python

    def __init__(self, config, data):
        super().__init__(config, data)
        self.emb_table = init_emb_table(self.config, self.info['num_nodes'])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.config['emb_dim'], 1024), 
            torch.nn.Tanh(), 
            torch.nn.Linear(1024, self.config['emb_dim']), 
        )
        self.opt = torch.optim.Adam([
            {'params': self.emb_table.parameters(),'lr': self.config['emb_lr']},
            {'params': self.mlp.parameters(), 'lr': self.config['dnn_lr']},
        ])


3. Implement forward_and_backward()
-----------------------------

.. code:: python

    def forward_and_backward(self, batch_data):
        ((src, pos, neg), ) = batch_data

        src_emb = self.mlp(self.emb_table(src))
        pos_emb = self.mlp(self.emb_table(pos))
        neg_emb = self.mlp(self.emb_table(neg))

        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)

        loss = bpr_loss(pos_score, neg_score)

        rw = self.config['L2_reg_weight']
        L2_reg_loss = 1/2 * (1 / len(src)) * (
            (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
        )
        loss += rw * L2_reg_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.item()


4. Implement on_eval_begin()
-----------------------------


.. code:: python

    @torch.no_grad()
    def on_eval_begin(self):
        self.out_emb_table = torch.empty(
            size=self.emb_table.weight.shape, dtype=torch.float32
        )
        dl = torch.utils.data.DataLoader(
            dataset=torch.arange(self.info['num_nodes']), 
            batch_size=256
        )
        for nids in tqdm(dl, desc="infer output emb"):
            self.out_emb_table[nids] = self.mlp(self.emb_table(nids))
        
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table


5. Add model to build_Model()
-----------------------------
