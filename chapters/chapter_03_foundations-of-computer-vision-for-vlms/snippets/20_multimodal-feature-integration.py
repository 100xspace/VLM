from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
# Load pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
# Load input image and candidate captions
img = Image.open("sample.jpg").convert("RGB")
text = ["a photo of a cat", "a photo of a dog"]
# Preprocess inputs
inputs = processor(text=text, images=img, return_tensors="pt", padding=True)
# Forward pass and similarity scores
with torch.no_grad():
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print(probs)  # Example output: tensor([[0.89, 0.11]])
