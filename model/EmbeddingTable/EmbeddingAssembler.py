import torch


class EmbeddingAssembler:
    
    def __init__(self, config, data):
        # build embedding tables
        self.id_emb_table  # one learnable embedding for each node
        self.feat_emb_table_list = []  # node feature embedding
        for _config in config['emb_table_config_list']:
            pass
            self.emb_table_list.append(None)
        
        # build MLP
        self.feat_MLP = None  # MLP for projecting feature embedding
        
    def __call__(self, nids):
        return self.forward(nids)
    
    def forward(self, nids):
        # get node ID embedding
        id_emb = self.id_emb_table(nids)
        
        # get feature embedding
        ## get embedding for each feature
        feat_emb_list = []
        for table in self.emb_table_list:
            feat_emb_list.append(table(nids))
        
        ## concat all the feature embeddings
        feat_emb = torch.cat(feat_emb_list, dim=-1)
        
        ## project feat_emb to the same length as the node ID embedding
        feat_emb = self.MLP(feat_emb)

        # merge id embedding and feature embedding
        emb = id_emb + feat_emb
        
        return emb
