def encode_anyres(images, vision_encoder, sep_token):
embeddings = []
    for img in images:
    emb = vision_encoder(img)
    embeddings.append(emb)
    embeddings.append(sep_token)
    return torch.cat(embeddings, dim=1)
