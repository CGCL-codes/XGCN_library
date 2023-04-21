Custmize DataLoader
=========================

In this section, we'll first present an overview of the modules in DataLoader and 
then customize a new one. 

1. Overview
-----------------------------

In XGCN, the dataloader is called by ``Trainer`` during the batch training, 
and basically it only needs to be an iterable object. 
To add a new dataloader, you can simply implement it as an iterable object
on your own and add it to ``XGCN.create_DataLoader()`` 
(see ``create_DataLoader()`` in ``XGCN/dataloading/create.py``), 
or you can use the infrastructure provided by XGCN. 

In this section, we focus on introducing main components of
the XGCN dataloader infrastructure. 
The UML class diagram is shown in the figure below. 

.. image:: ../asset/dataloader_arch.jpg
  :width: 600
  :alt: UML class diagram of DataLoader

The interface classes are defined in ``XGCN/dataloading/base.py``, they describe 
a series of interface functions. 
The ``BaseDataset`` class requires three functions: ``__len__()``, ``__getitem__()``, 
and ``on_epoch_start()``. 
Note that the ``__getitem__()`` function is supposed to return a batch of training sample 
given the batch index. 
``NodeListDataset`` further requires the returned data in ``__getitem__()`` 
should include a list of tensors of node IDs. 
The ``Sampler`` is used to generate positive/negative training samples 
given the sample index. And the ``BatchSampleIndicesGenerator`` is used to 
generate sample indices given the batch index. 

To train large-scale message-passing GNNs, mini-graph sampling is often needed. 
In XGCN, the ``BlockDataset`` class utilize the ``NodeListDataset``
and DGL's ``BlockSampler`` to conduct mini-graph sampling. 

The function ``XGCN.create_DataLoader()`` is used to initialize a dataloader. 
You can refer to the functions in ``XGCN/dataloading/create.py``. 


2. Implement a Sampler
-----------------------------

In the following, let's customize a new dataloader and apply it to a GNN model. 
Suppose we want to sample negative nodes according to their degrees, this can be 
done by adding a new Sampler (add a ``XGCN/dataloading/sample/WeightedNeg_Sampler.py``): 

.. code:: python
    
    # XGCN/dataloading/sample/WeightedNeg_Sampler.py

    from XGCN.dataloading.base import BaseSampler
    from XGCN.utils import io, csr

    import torch
    import os.path as osp


    class WeightedNeg_Sampler(BaseSampler):
        
        def __init__(self, config, data):
            self.num_neg = config['num_neg']
            
            data_root = config['data_root']
            indptr = io.load_pickle(osp.join(data_root, 'indptr.pkl'))
            indices = io.load_pickle(osp.join(data_root, 'indices.pkl'))
            indptr, indices = csr.get_undirected(indptr, indices)
            degrees = indptr[1:] - indptr[:-1]
            
            info = io.load_yaml(osp.join(data_root, 'info.yaml'))
            if info['graph_type'] == 'user-item':
                self.num_neg_total = info['num_items']
                self.offset = info['num_users']
            else:
                self.num_neg_total = info['num_nodes']
                self.offset = 0
            
            # the probability a node is sampled is proportional to the weights:
            self.weights = torch.FloatTensor(
                degrees[self.offset : self.offset + self.num_neg_total]
            ) ** 0.75
            
        def __call__(self, pos_sample_data):
            src = pos_sample_data['src']
            neg = torch.multinomial(
                self.weights, num_samples=len(src), replacement=True
            ) + self.offset
            return neg


Also remember to add it to the ``create_Sampler()`` function in ``XGCN/dataloading/create.py``, 
so that XGCN can find the new Sampler.

.. code:: python
    
    # in XGCN/dataloading/create.py

    from XGCN.sample.WeightedNeg_Sampler import WeightedNeg_Sampler

    def create_LinkDataset(config, data):
        pos_sampler = {
            'ObservedEdges_Sampler': ObservedEdges_Sampler,
        }[config['pos_sampler']](config, data)
        
        neg_sampler = {
            'RandomNeg_Sampler': RandomNeg_Sampler,
            'WeightedNeg_Sampler': WeightedNeg_Sampler,  # <-- add the new Sampler here
        }[config['neg_sampler']](config, data)

        ...


3. Config and run!
-----------------------------

Now we have already add a new dataloader to XGCN, you can use it by simply 
add a ``--neg_sampler WeightedNeg_Sampler`` argument. 
For example, we can modify the script: ``XGCN/script/model/GraphSAGE/run_GraphSGAE-facebook.sh``.

.. code:: bash
    
    # in XGCN/script/model/GraphSAGE/run_GraphSGAE-facebook.sh

    python -m XGCN.main.run_model --seed $seed \
        ...
        --neg_sampler WeightedNeg_Sampler \
