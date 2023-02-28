import torch
import torch.nn.functional as F


def bpr_loss(pos_score, neg_score):
    if pos_score.shape != neg_score.shape:
        num_neg = neg_score.shape[-1]
        pos_score = pos_score.repeat_interleave(
            num_neg, dim=-1).reshape(neg_score.shape)
        
    return torch.mean(F.softplus(neg_score - pos_score))


def bce_loss(pos_score, neg_score, neg_weight=1.0, reduction='mean', pos_coe=None, neg_coe=None):
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
