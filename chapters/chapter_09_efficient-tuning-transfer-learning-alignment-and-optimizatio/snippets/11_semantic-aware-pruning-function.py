def semantic_prune(tokens, attn_weights, keep_ratio=0.5):
    """
tokens: (B, N, D)
attn_weights: (B, H, N, N)
"""
entropy = token_entropy(attn_weights)
num_keep = int(tokens.size(1) * keep_ratio)
_, keep_idx = torch.topk(entropy, num_keep, dim=1)
pruned_tokens = torch.stack([
    tokens[b, keep_idx[b]] for b in range(tokens.size(0))
])
    return pruned_tokens
