from gnn_zoo.model.xGCN import xGCN
from gnn_zoo.model.GensimNode2vec import GensimNode2vec
from gnn_zoo.model.GraphSAGE import GraphSAGE
from gnn_zoo.model.GAT import GAT
from gnn_zoo.model.GIN import GIN
from gnn_zoo.model.LightGCN import LightGCN
from gnn_zoo.model.PPRGo import PPRGo
from gnn_zoo.model.UltraGCN import UltraGCN


def build_Model(config, data):
    if config['model'] == 'Node2vec':
        from gnn_zoo.model.Node2vec import Node2vec
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
