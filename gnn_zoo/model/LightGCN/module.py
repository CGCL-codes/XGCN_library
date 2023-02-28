import torch


def get_lightgcn_out_emb(A, base_emb_table, num_gcn_layers, stack_layers=True):
    if not stack_layers:
        X = base_emb_table
        for _ in range(num_gcn_layers):
            X = A @ X
        return X
    else:
        X_list = [base_emb_table]
        for _ in range(num_gcn_layers):
            X = A @ X_list[-1]
            X_list.append(X)
        X_out = torch.stack(X_list, dim=1).mean(dim=1)
    return X_out
