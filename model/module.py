import dgl
import torch
import torch.nn.functional as F


def dot_product(src_emb, dst_emb):
    if src_emb.shape != dst_emb.shape:
        return (src_emb.unsqueeze(-2) * dst_emb).sum(dim=-1)
    else:
        return (src_emb * dst_emb).sum(dim=-1)


def bpr_loss(pos_score, neg_score):
    if pos_score.shape != neg_score.shape:
        num_neg = neg_score.shape[-1]
        pos_score = pos_score.repeat_interleave(
            num_neg, dim=-1).reshape(neg_score.shape)
        
    return torch.mean(torch.nn.functional.softplus(neg_score - pos_score))


def bce_loss(pos_score, neg_score, neg_weight, reduction='mean', pos_coe=None, neg_coe=None):
    device = pos_score.device
    pos_loss = F.binary_cross_entropy_with_logits(
        pos_score, 
        torch.ones(pos_score.shape).to(device),
        weight=pos_coe
    )
    neg_loss = F.binary_cross_entropy_with_logits(
        neg_score, 
        torch.zeros(neg_score.shape).to(device),
        weight=neg_coe
    ).mean(dim=-1)
    
    loss = pos_loss + neg_weight * neg_loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    
    return loss


_cosine_similarity = torch.nn.CosineSimilarity(dim=-1)


def cosine_contrastive_loss(src_emb, pos_emb, neg_emb, margin, neg_weight=None):
    pos_loss = 1 - _cosine_similarity(src_emb, pos_emb)
    
    if src_emb.shape != neg_emb.shape:
        src_emb = src_emb.unsqueeze(-2)
    neg_loss = torch.relu(_cosine_similarity(src_emb, neg_emb) - margin)
    
    if neg_weight is None:
        loss = (pos_loss + neg_loss.sum(dim=-1)).mean()
    else:
        loss = (pos_loss + neg_weight * neg_loss.mean(dim=-1)).mean()
    return loss


def ssm_loss(emb1, emb2, emb3=None, tao=0.2):
    # shape of emb1, emb2: (batch_size, emb_dim)
    # shape of emb3: (batch_size, k, emb_dim)
    
    # for numerator in InfoNCE loss
    nu = _cosine_similarity(emb1, emb2)  # shape: (batch_size, )
    batch_size = len(nu)
    
    # for denominator in InfoNCE loss
    if emb3 is None:
        de = _cosine_similarity(  # shape: (batch_size, batch_size)
            emb1.repeat_interleave(repeats=batch_size, dim=0),  # [a0, a0, a0, a1, a1, a1, ...]
            emb2.repeat(batch_size, 1)                          # [b0, b1, b2, b0, b1, b2, ...]
        ).reshape(-1, batch_size)
    else:
        de = _cosine_similarity(emb1.unsqueeze(-2), emb3)
    
    nu = (nu / tao).exp()              # shape: (batch_size, )
    de = (de / tao).exp().sum(dim=-1)  # shape: (batch_size, )
    
    loss = -(nu / de).log().mean()
    return loss
