import torch
import clip
from PIL import Image

# Load pretrained CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load a sample image and candidate text labels
image = preprocess(Image.open("sample_dog.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a photo of a dog", "a photo of a cat"]).to(device)
