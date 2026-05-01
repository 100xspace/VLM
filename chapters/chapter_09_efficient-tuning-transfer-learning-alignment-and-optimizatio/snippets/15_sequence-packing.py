def pack_sequences(sequences, max_len=4096):
packed = []
current = []
    for seq in sequences:
        if sum(len(s) for s in current) + len(seq) > max_len:
        packed.append(torch.cat(current))
        current = []
    current.append(seq)
    if current:
    packed.append(torch.cat(current))
    return packed
