def naive_attention_prune(tokens, attn_weights, keep_ratio=0.5):
    """
tokens: (B, N, D)
attn_weights: (B, H, N, N)
"""
    # Average attention over heads and queries
importance = attn_weights.mean(dim=(1, 2))
num_keep = int(tokens.size(1) * keep_ratio)
_, keep_idx = torch.topk(importance, num_keep, dim=1)
pruned = torch.stack([
    tokens[b, keep_idx[b]] for b in range(tokens.size(0))
])
    return pruned
