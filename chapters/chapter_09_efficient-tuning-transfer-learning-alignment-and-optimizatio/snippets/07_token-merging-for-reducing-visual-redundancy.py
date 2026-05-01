def tome_merge(tokens, merge_ratio=0.5):
    """
tokens: (B, N, D)
merge_ratio: fraction of tokens to remove
"""
B, N, D = tokens.shape
num_merge = int(N * merge_ratio)
    # Normalize tokens for cosine similarity
tokens_norm = F.normalize(tokens, dim=-1)
    # Cosine similarity matrix: (B, N, N)
sim = torch.matmul(tokens_norm, tokens_norm.transpose(1, 2))
    # Remove self-similarity
eye = torch.eye(N, device=tokens.device).bool()
sim.masked_fill_(eye.unsqueeze(0), -1)
    # Flatten similarity and select top pairs
sim_flat = sim.view(B, -1)
_, idx = torch.topk(sim_flat, num_merge, dim=-1)
merged_tokens = tokens.clone()
    for b in range(B):
    used = set()
        for index in idx[b]:
        i = index // N
        j = index % N
            if i.item() in used or j.item() in used:
