import torch
import torch.nn as nn


class RefNet(torch.nn.Module):
        
    def __init__(self, dnn_arch, scale_net_arch=None):
        super(RefNet, self).__init__()
        self.dnn = torch.nn.Sequential(*eval(dnn_arch))
        # self.dnn = torch.nn.Sequential(
        #     torch.nn.Linear(64, 1024), 
        #     torch.nn.Tanh(), 
        #     torch.nn.Linear(1024, 1024), 
        #     torch.nn.Tanh(), 
        #     torch.nn.Linear(1024, 64)
        # )
        
        if scale_net_arch is not None:
            self.scale_net = torch.nn.Sequential(*eval(scale_net_arch))
        else:
            self.scale_net = None
        # self.scale_net = torch.nn.Sequential(
        #     torch.nn.Linear(64, 32),
        #     torch.nn.Tanh(),
        #     torch.nn.Linear(32, 1),
        #     torch.nn.Sigmoid()
        # )
    
    def forward(self, X):
        if self.scale_net is not None:
            theta = self.scale_net(X)
            X = theta * self.dnn(X)
        else:
            X = self.dnn(X)
        return X
