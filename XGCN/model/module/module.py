def dot_product(src_emb, dst_emb):
    if src_emb.shape != dst_emb.shape:
        return (src_emb.unsqueeze(-2) * dst_emb).sum(dim=-1)
    else:
        return (src_emb * dst_emb).sum(dim=-1)
