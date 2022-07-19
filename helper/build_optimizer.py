import torch

def build_optimizer(config, data):
    model = data['model']
    
    if config['model'] == 'node2vec':
        print("## use SparseAdam")
        opt = torch.optim.SparseAdam(model.model.parameters(), lr=config['emb_lr'])
    else:
        if config['use_sparse']:
            print("## use SparseAdam")
            opt = torch.optim.SparseAdam(model.parameters())
        else:
            opt = torch.optim.Adam(model.parameters())
    return opt
