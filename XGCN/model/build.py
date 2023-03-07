from XGCN.model.xGCN import xGCN
from XGCN.model.GensimNode2vec import GensimNode2vec
from XGCN.model.GraphSAGE import GraphSAGE
from XGCN.model.GAT import GAT
from XGCN.model.GIN import GIN
from XGCN.model.LightGCN import LightGCN
from XGCN.model.PPRGo import PPRGo
from XGCN.model.UltraGCN import UltraGCN
from XGCN.model.SGC import SGC, SGC_learnable_emb
from XGCN.model.SSGC import SSGC, SSGC_learnable_emb
from XGCN.model.SIGN import SIGN, SIGN_learnable_emb
from XGCN.model.GBP import GBP
from XGCN.model.GAMLP import GAMLP, GAMLP_learnable_emb


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
            'SGC': SGC,
            'SGC_learnable_emb': SGC_learnable_emb,
            'SSGC': SSGC,
            'SSGC_learnable_emb': SSGC_learnable_emb,
            'SIGN': SIGN,
            'SIGN_learnable_emb': SIGN_learnable_emb,
            'GBP': GBP,
            'GAMLP': GAMLP,
            'GAMLP_learnable_emb': GAMLP_learnable_emb,
        }[config['model']](config, data)
    return model
