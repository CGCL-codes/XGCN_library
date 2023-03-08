Custmize DataLoader
=========================

In this section, we'll first present an overview of the modules in DataLoader and 
then customize a new one. 

1. Overview
-----------------------------

In XGCN, the dataloader is called by ``Trainer`` during the batch training, 
and basically it only needs to be an iterable object. 
To add a new dataloader, you can simply implement it as an iterable object
on your own and add it to ``XGCN.build_DataLoader()`` 
(see ``build_DataLoader()`` in ``XGCN/dataloading/build.py``), 
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

The function ``XGCN.build_DataLoader()`` is used to initialize a dataloader. 
You can refer to the functions in ``XGCN/dataloading/build.py``. 


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


Also remember to add it to the ``build_Sampler()`` function in ``XGCN/dataloading/build.py``, 
so that XGCN can find the new Sampler.

.. code:: python
    
    # in XGCN/dataloading/build.py

    from XGCN.sample.WeightedNeg_Sampler import WeightedNeg_Sampler

    def build_Sampler(sampler_type, config, data):
        sampler = {
            'ObservedEdges_Sampler': ObservedEdges_Sampler,
            'RandomNeg_Sampler': RandomNeg_Sampler,
            'WeightedNeg_Sampler': WeightedNeg_Sampler,  # <-- add the new Sampler here
        }[sampler_type](config, data)
        return sampler


3. Config and run!
-----------------------------

.. code:: bash
    
    # write your own paths here:
    all_data_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/XGCN_data
    config_file_root=/media/xreco/DEV/xiran/code/XGCN_package_dev/xGCN/config

    dataset=facebook
    model=GraphSAGE
    seed=0

    data_root=$all_data_root/dataset/instance_$dataset
    results_root=$all_data_root/model_output/$dataset/$model/[seed$seed]

    file_pretrained_emb=$all_data_root/model_output/$dataset/Node2vec/[seed$seed]/out_emb_table.pt

    python -m XGCN.main.run_model --seed $seed \
        --config_file $config_file_root/$model-config.yaml \
        --data_root $data_root --results_root $results_root \
        --val_evaluator WholeGraph_MultiPos_Evaluator --val_batch_size 256 \
        --file_val_set $data_root/val_set.pkl \
        --test_evaluator WholeGraph_MultiPos_Evaluator --test_batch_size 256 \
        --file_test_set $data_root/test_set.pkl \
        --from_pretrained 1 \
        --file_pretrained_emb $file_pretrained_emb \
        --freeze_emb 0 \
        --neg_sampler WeightedNeg_Sampler \
