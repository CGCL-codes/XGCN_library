class BaseDataset(object):
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, batch_idx):
        raise NotImplementedError

    def on_epoch_start(self):
        raise NotImplementedError


class NodeListDataset(BaseDataset):

    def __getitem__(self, batch_idx):
        # node_list = []
        # other_data = None
        # return [node_list, other_data]
        raise NotImplementedError


class BaseSampler(object):
    
    def __call__(self, **kwargs):
        raise NotImplementedError


class BatchSampleIndicesGenerator(object):

    def __len__(self):
        # return num_batch_per_epoch
        raise NotImplementedError
    
    def __getitem__(self, batch_idx):
        # return batch_sample_indices
        raise NotImplementedError
    
    def on_epoch_start(self):
        raise NotImplementedError
