from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained LLaVA model and processor
model_id = "liuhaotian/LLaVA-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

# Load a sample image and prepare a visual query
image = Image.open("sample_kitchen.jpg")
prompt = "Describe the scene and the main objects visible."
