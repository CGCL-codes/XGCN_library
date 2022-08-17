from model.Node2vecWrapper import Node2vecWrapper
from model.LightGCN import LightGCN
from model.xGCN import xGCN
from model.xGCN_res import xGCN_res
from model.xGCN_multi import xGCN_multi
from model.xxGCN import xxGCN
from model.UltraGCN.UltraGCN import UltraGCN
from model.UltraGCN.UltraGCN_v0 import UltraGCN_v0
from model.SimpleX import SimpleX
from model.PPRGo import PPRGo
from model.GraphSAGE import GraphSAGE
from model.GAT import GAT
from model.GIN import GIN
from model.Block_LightGCN import Block_LightGCN
from model.Block_SimpleX import Block_SimpleX


def build_model(config, data):
    model = {
        'node2vec': Node2vecWrapper,
        'lightgcn': LightGCN,
        'block_lightgcn': Block_LightGCN,
        'block_simplex': Block_SimpleX,
        'xgcn': xGCN,
        'xgcn_res': xGCN_res,
        'xgcn_multi': xGCN_multi,
        'xxgcn': xxGCN,
        'ultragcn': UltraGCN,
        'ultragcn_v0': UltraGCN_v0,
        'simplex': SimpleX,
        'pprgo': PPRGo,
        'graphsage': GraphSAGE,
        'gat': GAT,
        'gin': GIN,
    }[config['model']](config, data)
    
    data['model'] = model
    
    return model
