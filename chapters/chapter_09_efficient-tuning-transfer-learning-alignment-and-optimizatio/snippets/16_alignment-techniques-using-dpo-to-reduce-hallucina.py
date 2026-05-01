def build_attention_mask(lengths):
masks = []
    for idx, l in enumerate(lengths):
    mask = torch.zeros(sum(lengths), sum(lengths))
    start = sum(lengths[:idx])
    mask[start:start+l, start:start+l] = 1
    masks.append(mask)
    return torch.stack(masks)

Note: the original lengths.index(l) call returned the first index of value l, which produced wrong offsets whenever two samples had identical lengths. Iterating with enumerate fixes this by using the actual position in the list.
