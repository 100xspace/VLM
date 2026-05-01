from torch.quantization import quantize_dynamic
# Apply dynamic INT8 quantization (CPU-optimized in standard PyTorch)
from torch.quantization import quantize_dynamic
# Utility: calculate model size in MB
def get_model_size_mb(model, tmp="temp_model.pth"):
    torch.save(model.state_dict(), tmp); size_mb = os.path.getsize(tmp) / 1024 / 1024; os.remove(tmp); return size_mb
# Quantize to INT8 (model must be on CPU for dynamic quantization)
model_int8_dynamic = quantize_dynamic(model_fp32.cpu(), {nn.Linear}, dtype=torch.qint8)
print("\n=== Dynamic INT8 Quantization ===")
fp32_size = get_model_size_mb(model_fp32.cpu()); int8_size = get_model_size_mb(model_int8_dynamic)
print(f"FP32 model size: {fp32_size:.2f} MB"); print(f"INT8 model size: {int8_size:.2f} MB"); print(f"Compression ratio: {fp32_size/int8_size:.2f}x")
# Quick CPU inference sanity check (not apples-to-apples vs GPU FP32 timing)
start = time.time()
with torch.no_grad():
    img = Image.open(BytesIO(requests.get(test_image_url).content)).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    outputs = model_int8_dynamic.generate(**inputs, max_length=50)
print(f"INT8 CPU Inference time: {(time.time()-start)*1000:.2f} ms")
print(f"INT8 Caption: {processor.decode(outputs[0], skip_special_tokens=True)}")
