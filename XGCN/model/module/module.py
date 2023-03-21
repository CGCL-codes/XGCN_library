import torch


def dot_product(src_emb, dst_emb):
    if src_emb.shape != dst_emb.shape:
        return (src_emb.unsqueeze(-2) * dst_emb).sum(dim=-1)
    else:
        return (src_emb * dst_emb).sum(dim=-1)


def cosine_contrastive_loss(src_emb, pos_emb, neg_emb, margin, neg_weight=None):
    _cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    pos_loss = 1 - _cosine_similarity(src_emb, pos_emb)
    
    if src_emb.shape != neg_emb.shape:
        src_emb = src_emb.unsqueeze(-2)
    neg_loss = torch.relu(_cosine_similarity(src_emb, neg_emb) - margin)
    
    if neg_weight is None:
        loss = (pos_loss + neg_loss.sum(dim=-1)).mean()
    else:
        loss = (pos_loss + neg_weight * neg_loss.mean(dim=-1)).mean()
    return loss
