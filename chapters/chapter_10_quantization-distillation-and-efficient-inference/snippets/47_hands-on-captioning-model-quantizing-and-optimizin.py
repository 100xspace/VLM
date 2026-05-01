import os, time, requests
import torch, torch.nn as nn
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from PIL import Image
from io import BytesIO
# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"; print(f"Using device: {device}")
# Load a small VLM for captioning (~180M params)
model_name = "microsoft/git-base-coco"
processor = AutoProcessor.from_pretrained(model_name)
model_fp32 = AutoModelForVision2Seq.from_pretrained(model_name).to(device).eval()
print(f"Model loaded: {model_name}"); print(f"Number of parameters: {sum(p.numel() for p in model_fp32.parameters())/1e6:.1f}M")
