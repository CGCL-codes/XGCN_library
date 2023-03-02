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

(5) ``load()``. This function is call by ``Trainer`` 


Specifically, these interface functions are described by the ``BaseModel`` class 
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


2. Implement __init__()
-----------------------------


XGCN provides a ``BaseEmbeddingModel`` class which is inherited from ``BaseModel`` 
and implements some useful functions for model evaluation. 
With the ``BaseEmbeddingModel`` class, we only need to implement two interface functions: 
``forward_and_backward()`` 
Let's start from it and create a new model. 


.. code:: python

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

3. Implement forward_and_backward()
-----------------------------

4. Implement on_eval_begin()
-----------------------------

5. Add model to build_Model()
-----------------------------
