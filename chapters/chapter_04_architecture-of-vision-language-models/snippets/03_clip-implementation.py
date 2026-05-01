with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)

# Compute cosine similarity
similarity = (100.0 * image_features @ text_features.T)
probs = similarity.softmax(dim=-1)
print("Probabilities:", probs.cpu().numpy())
