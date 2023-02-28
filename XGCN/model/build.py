from XGCN.model.xGCN import xGCN
from XGCN.model.GensimNode2vec import GensimNode2vec
from XGCN.model.GraphSAGE import GraphSAGE
from XGCN.model.GAT import GAT
from XGCN.model.GIN import GIN
from XGCN.model.LightGCN import LightGCN
from XGCN.model.PPRGo import PPRGo
from XGCN.model.UltraGCN import UltraGCN


def build_Model(config, data):
    if config['model'] == 'Node2vec':
        from XGCN.model.Node2vec import Node2vec
        model = Node2vec(config, data)
    else:
        model = {
            'xGCN': xGCN,
            'GensimNode2vec': GensimNode2vec,
            'GraphSAGE': GraphSAGE,
            'GAT': GAT,
            'GIN': GIN,
            'LightGCN': LightGCN,
            'PPRGo': PPRGo,
            'UltraGCN': UltraGCN,
        }[config['model']](config, data)
    return model
