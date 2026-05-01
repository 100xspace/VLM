used.add(i.item())
        used.add(j.item())
    # Keep only unmerged tokens
keep_mask = torch.ones(N, dtype=torch.bool)
keep_mask[list(used)] = False
    return merged_tokens[:, keep_mask]

Note: the nested Python loop above is intentionally explicit for readability. On realistic token counts (N in the thousands) it becomes a bottleneck; production implementations use fully vectorized bipartite matching on the GPU and avoid the per-batch Python loop entirely.
