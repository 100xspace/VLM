from transformers import BitsAndBytesConfig
# GPU 4-bit quantization (NF4) is the common production path for large models
def load_4bit_model(model_name):
    # Configuration for 4-bit quantization (requires bitsandbytes)
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    # Load model directly to GPU in 4-bit
    return AutoModelForVision2Seq.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto").eval()
print("\n=== GPU 4-bit (NF4) Quantization ===")
try:
    model_4bit = load_4bit_model(model_name)
    # Measure VRAM usage
    vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024; print(f"4-bit Model VRAM usage: {vram_mb:.2f} MB")
    # Run inference
    img = Image.open(BytesIO(requests.get(test_image_url).content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to("cuda")
    start = time.time(); outputs = model_4bit.generate(**inputs, max_length=50); end = time.time()
    print(f"4-bit Inference time: {(end-start)*1000:.2f} ms")
    print(f"4-bit Caption: {processor.decode(outputs[0], skip_special_tokens=True)}")
except ImportError:
    print("bitsandbytes not installed. Skipping 4-bit demonstration.")
except Exception as e:
    print(f"4-bit quantization failed: {e}")
