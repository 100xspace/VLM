def token_entropy(attn):
    """
attn: (B, H, N, N)
"""
p = attn.mean(dim=1)
entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1)
    return entropy
