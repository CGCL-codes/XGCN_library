DataLoader
=========================


1. Interface Classes of the DataLoader
----------------------------------------

The UML class diagram of the dataloader designing is shown in the figure below.

.. image:: ../asset/dataloader_arch.jpg
  :width: 600
  :alt: UML class diagram of DataLoader designing


2. LinkDataset
----------------------------------------


.. code:: python

    class LinkDataset(NodeListDataset):
        
        def __init__(self,
                    pos_sampler: BaseSampler,
                    neg_sampler: BaseSampler,
                    batch_sample_indices_generator: BatchSampleIndicesGenerator):
            self.pos_sampler = pos_sampler
            self.neg_sampler = neg_sampler
            self.batch_sample_indices_generator = batch_sample_indices_generator

        def __len__(self):
            return len(self.batch_sample_indices_generator)
        
        def __getitem__(self, batch_idx):
            batch_sample_indices = self.batch_sample_indices_generator[batch_idx]
            
            src, pos = self.pos_sampler(batch_sample_indices)
            neg = self.neg_sampler({'src': src, 'pos': pos})
            
            node_list = [src, pos, neg]
            return [node_list, ]
        
        def on_epoch_start(self):
            self.batch_sample_indices_generator.on_epoch_start()


3. BlockDataset
----------------------------------------

.. code:: python

    class BlockDataset(BaseDataset):
        
        def __init__(self, g, block_sampler,
                    dataset: NodeListDataset):
            self.g = g
            self.block_sampler = block_sampler
            self.dataset = dataset
            
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, batch_idx):
            re = self.dataset[batch_idx]
            node_list = re[0]
            
            flat_batch_nid = torch.cat([nid.flatten() for nid in node_list])
            
            unique_nid, idx_mapping = get_unique(flat_batch_nid)
            # unique_nid: 1d-tensor, remove repeated nid in batch_nid
            # idx_mapping: map a nid into the index of the unique_nid tensor

            input_nid, output_nid, blocks = self.block_sampler.sample_blocks(
                self.g, unique_nid.to(self.g.device), exclude_eids=None
            )
            
            return re, (input_nid, output_nid, blocks, idx_mapping)

        def on_epoch_start(self):
            self.dataset.on_epoch_start()
