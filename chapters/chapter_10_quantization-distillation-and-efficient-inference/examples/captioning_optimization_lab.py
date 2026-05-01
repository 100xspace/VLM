"""Hands-on captioning optimization lab.

This script loads a small image captioning model, runs a baseline pass, and can
optionally apply CPU dynamic quantization to the language head layers.
"""

from __future__ import annotations

import argparse
import time
from io import BytesIO

import requests
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


DEFAULT_IMAGE_URL = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        response = requests.get(path_or_url, timeout=20)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def generate_caption(model, processor, image: Image.Image, device: str, max_new_tokens: int = 24) -> tuple[str, float]:
    inputs = processor(images=image, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed_ms = (time.perf_counter() - start) * 1000
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return caption, elapsed_ms


def dynamic_quantize_for_cpu(model):
    return torch.quantization.quantize_dynamic(model.cpu(), {torch.nn.Linear}, dtype=torch.qint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact VLM captioning optimization lab.")
    parser.add_argument("--model", default="microsoft/git-base-coco")
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--dynamic-quant", action="store_true", help="Apply CPU dynamic INT8 quantization.")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    device = "cpu" if args.dynamic_quant else ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForVision2Seq.from_pretrained(args.model).to(device).eval()
    if args.dynamic_quant:
        model = dynamic_quantize_for_cpu(model).eval()

    image = load_image(args.image)
    caption, elapsed_ms = generate_caption(model, processor, image, device=device, max_new_tokens=args.max_new_tokens)
    print({"device": device, "dynamic_quant": args.dynamic_quant, "latency_ms": round(elapsed_ms, 2), "caption": caption})


if __name__ == "__main__":
    main()
