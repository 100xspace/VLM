import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load the Medical VLM
# 'flaviagiammarino/pubmed-clip-vit-base-patch32' is a popular specific implementation
model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)

def predict_pneumonia_vlm(image_path):
    # 2. Define Classes via Text Prompts
    prompts = [
        "A chest X-ray of a patient with pneumonia",
        "A chest X-ray of a healthy patient"
    ]

    # 3. Load and Process Image
    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    )

    # 4. Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Calculate Probabilities
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    pneumonia_prob = probs[0][0].item()

    return pneumonia_prob

# Usage
# prob = predict_pneumonia_vlm("path/to/xray.png")
# print(f"VLM Confidence: {prob:.2%}")
