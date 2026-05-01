def vispruner(tokens, saliency_map, similarity_thresh=0.95):
    """
tokens: (B, N, D)
saliency_map: (B, N)
"""
tokens_norm = F.normalize(tokens, dim=-1)
sim = torch.matmul(tokens_norm, tokens_norm.transpose(1, 2))
keep_mask = saliency_map > saliency_map.mean(dim=1, keepdim=True)
    # Remove highly redundant low-saliency tokens
    for b in range(tokens.size(0)):
        for i in range(tokens.size(1)):
            if not keep_mask[b, i]:
                if (sim[b, i] > similarity_thresh).any():
                keep_mask[b, i] = False
    return tokens[keep_mask].view(tokens.size(0), -1, tokens.size(-1))
