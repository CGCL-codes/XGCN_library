import torch

def build_optimizer(config, data):
    model = data['model']
    
    if config['model'] == 'node2vec':
        opt = torch.optim.SparseAdam(list(model.model.parameters()), lr=config['emb_lr'])
    else:
        opt = torch.optim.Adam(model.parameters())
    return opt
