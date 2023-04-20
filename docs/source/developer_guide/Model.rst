Customize Model
=========================

In this part, let's dive into the implementations of a model 
by customizing a new one. 
We'll first introduce the interface functions which are supposed to be implemented, 
and then give a model implementation example. 


1. Interface functions
-----------------------------

XGCN provides a ``BaseEmbeddingModel`` class, and all the embedding models should 
inherit it. To get started, let's create a ``XGCN/model/NewModel.py`` with the following 
contents: 

.. code:: python

    from XGCN.model.base import BaseEmbeddingModel
    from XGCN.model.module import init_emb_table, dot_product, bce_loss
    import torch
    
    class NewModel(BaseEmbeddingModel):
        
        def __init__(self, config):
            super().__init__(config)
            
        def forward_and_backward(self, batch_data):
            pass
        
        @torch.no_grad()
        def infer_out_emb_table(self):
            pass

        def save(self, root=None):
            pass
        
        def load(self, root=None):
            pass

There are 5 interface functions to be implemented: 

(1) ``__init__()``: initialize model parameters and optimizers. 

(2) ``forward_and_backward()``: receive a batch training data, perform forward calculation and backward calculation (updating model parameters). 

(3) ``infer_out_emb_table()``: infer the output embeddings for all the nodes. 

(4) ``save()``: save the model and the optimizer state. 

(5) ``load()``: load the model and the optimizer state. 


2. Implement __init__()
-----------------------------

The ``__init__()`` function is responsible for initializing model parameters and optimizers. 
For simplicity, here we just create an embedding table, an MLP, and optimizers: 

.. code:: python

    def __init__(self, config):
        super().__init__(config)
        self.device = self.config['device']

        self.emb_table = init_emb_table(self.config, self.info['num_nodes'])
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.config['emb_dim'], 1024), 
            torch.nn.Tanh(), 
            torch.nn.Linear(1024, self.config['emb_dim']), 
        ).to(self.device)
        
        self.optimizers = {}
        if not self.config['freeze_emb']:
            if self.config['use_sparse']:  # use SparseAdam
                self.optimizers['emb_table-SparseAdam'] = torch.optim.SparseAdam(
                    [{'params':list(self.emb_table.parameters()), 
                        'lr': self.config['emb_lr']}]
                )
            else:
                self.optimizers['emb_table-Adam'] = torch.optim.Adam(
                    [{'params': self.emb_table.parameters(),
                        'lr': self.config['emb_lr']}]
                )
        self.optimizers['mlp-Adam'] = torch.optim.Adam(
            [{'params': self.mlp.parameters(), 'lr': self.config['dnn_lr']}]
        )

Note that the ``self.emb_table`` and ``self.optimizers`` objects 
are required by the ``BaseEmbeddingModel`` class. Please initialize them as above. 

If you have some new configuration arguments names, please add them 
in the ``_parse_arguments()`` function in ``XGCN/utils/parse_arguments.py``. 


3. Implement forward_and_backward()
-------------------------------------

The ``forward_and_backward()`` function receives batch training data, 
executes forward calculation, and performs backward propagation. 
Here we use the BCE loss and the L2 regularization: 

.. code:: python

    def forward_and_backward(self, batch_data):
        ((src, pos, neg), ) = batch_data

        src_emb = self.mlp(self.emb_table(src.to(self.device)))
        pos_emb = self.mlp(self.emb_table(pos.to(self.device)))
        neg_emb = self.mlp(self.emb_table(neg.to(self.device)))

        pos_score = dot_product(src_emb, pos_emb)
        neg_score = dot_product(src_emb, neg_emb)

        loss = bce_loss(pos_score, neg_score)

        rw = self.config['L2_reg_weight']
        L2_reg_loss = 1/2 * (1 / len(src)) * (
            (src_emb**2).sum() + (pos_emb**2).sum() + (neg_emb**2).sum()
        )
        loss += rw * L2_reg_loss

        self._backward(loss)  # the _backward function is already implemented by BaseEmbeddingModel

        return loss.item()    # need to return the loss value


4. Implement infer_out_emb_table()
------------------------------------

``infer_out_emb_table()`` specifies a ``self.out_emb_table`` and a ``self.target_emb_table`` 
that must be inferred in ``infer_out_emb_table()``. The former contains the output embeddings for 
all the nodes. And the latter is the embedding table for target nodes (e.g. in user-item graphs, 
the target nodes are items). 

.. code:: python

    @torch.no_grad()
    def infer_out_emb_table(self):
        self.out_emb_table = torch.empty(
            size=self.emb_table.weight.shape, dtype=torch.float32
        ).to(self.device)
        dl = torch.utils.data.DataLoader(
            dataset=torch.arange(self.info['num_nodes']), 
            batch_size=256
        )
        for nids in dl:
            self.out_emb_table[nids] = self.mlp(self.emb_table(nids.to(self.device)))
        
        if self.graph_type == 'user-item':
            self.target_emb_table = self.out_emb_table[self.info['num_users'] : ]
        else:
            self.target_emb_table = self.out_emb_table


5. Implement save() and load()
---------------------------------

The ``save()``\/``load()`` function are supposed to save/load the whole model and optimizers: 

.. code:: python

    def save(self, root=None):
        if root is None:
            root = self.model_root  # the self.model_root is set in BaseEmbeddingModel
        torch.save(self.mlp.state_dict(), osp.join(root, 'mlp-state_dict.pt'))
        self._save_optimizers(root)     # already implemented by BaseEmbeddingModel
        self._save_emb_table(root)
        self._save_out_emb_table(root)
    
    def load(self, root=None):
        if root is None:
            root = self.model_root
        self.mlp.load_state_dict(
            torch.load(osp.join(root, 'mlp-state_dict.pt'))
        )
        self._load_optimizers(root)     # already implemented by BaseEmbeddingModel
        self._load_emb_table(root)
        self._load_out_emb_table(root)


6. Add to build_Model()
---------------------------------

Once the model is complete, it should be added into ``XGCN.create_model()`` 
so that XGCN is able to find it: 

.. code:: python

    # XGCN/model/create.py
    from XGCN.model.xGCN import xGCN
    ...
    from XGCN.model.NewModel import NewModel  # <-- import your NewModel here

    def create_model(config, data):
        model = {
            'NewModel': NewModel,  # <-- add your NewModel here
            'xGCN': xGCN,
            ...
        }[config['model']](config)
        return model


7. Config and Run!
-----------------------------

Now we are ready to run the model, but before that, let's first 
make a template configuration file to make all the configuration arguments clear. 
For example, add a file - ``NewModel-config.yaml`` - in ``config/`` 
with the following contents: 

.. code:: yaml

    # config/NewModel-config.yaml
    # Dataset/Results root
    data_root: ""
    results_root: ""

    # Trainer configuration
    epochs: 200
    use_validation_for_early_stop: 1
    val_freq: 1
    key_score_metric: r20
    convergence_threshold: 20
    val_method: ""
    val_batch_size: 256
    file_val_set: ""

    # Testing configuration
    test_method: ""
    test_batch_size: 256
    file_test_set: ""

    # DataLoader configuration
    Dataset_type: NodeListDataset
    num_workers: 1
    NodeListDataset_type: LinkDataset
    pos_sampler: ObservedEdges_Sampler
    neg_sampler: RandomNeg_Sampler
    num_neg: 1
    BatchSampleIndicesGenerator_type: SampleIndicesWithReplacement
    train_batch_size: 1024
    str_num_total_samples: num_edges
    epoch_sample_ratio: 0.1

    # Model configuration
    model: NewModel
    seed: 1999
    device: "cuda:0"
    from_pretrained: 0
    file_pretrained_emb: ""
    freeze_emb: 0
    use_sparse: 1
    emb_dim: 64 
    emb_init_std: 0.1
    emb_lr: 0.005
    L2_reg_weight: 0.0

With the ``.yaml`` file, we can run the new model with the following script:

.. code:: bash

    # write your own paths here:
    all_data_root='/.../XGCN_data'
    config_file='../config/NewModel-config.yaml'
    
    python -m XGCN.main.run_model \
        --config_file $config_file \
        --data_root $all_data_root/dataset/instance_facebook \
        --results_root $all_data_root/model_output/NewModel \
        --file_val_set $all_data_root/dataset/val_set.pkl \
        --file_test_set $all_data_root/dataset/test_set.pkl \
