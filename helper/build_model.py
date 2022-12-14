from model.LightGCN import LightGCN
from model.xGCN import xGCN
from model.xGCN_ii import xGCN_ii
from model.xGCN_multi import xGCN_multi
from model.xxGCN import xxGCN
from model.UltraGCN.UltraGCN import UltraGCN
from model.UltraGCN.UltraGCN_v0 import UltraGCN_v0
from model.SimpleX import SimpleX
from model.PPRGo import PPRGo
from model.GraphSAGE import GraphSAGE
from model.GAT import GAT
from model.GIN import GIN
from model.SIGN import SIGN, SIGN_learnable_emb
from model.GAMLP import GAMLP
from model.Block_LightGCN import Block_LightGCN
from model.Block_SimpleX import Block_SimpleX


def build_model(config, data):
    model_dict = {
        'lightgcn': LightGCN,
        'block_lightgcn': Block_LightGCN,
        'block_simplex': Block_SimpleX,
        'xgcn': xGCN,
        'xgcn_ii': xGCN_ii,
        'xgcn_multi': xGCN_multi,
        'xxgcn': xxGCN,
        'ultragcn': UltraGCN,
        'ultragcn_v0': UltraGCN_v0,
        'simplex': SimpleX,
        'pprgo': PPRGo,
        'graphsage': GraphSAGE,
        'gat': GAT,
        'gin': GIN,
        'sign': SIGN,
        'sign_learnable_emb': SIGN_learnable_emb,
        'gamlp': GAMLP,
    }
    
    try:
        from model.Node2vecWrapper import Node2vecWrapper
        model_dict['node2vec'] = Node2vecWrapper
    except:
        pass
    
    model = model_dict[config['model']](config, data)
    
    data['model'] = model
    
    return model
