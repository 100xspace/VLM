def measure_inference_time(model, processor, image_url, num_runs=20):
    # Load image
    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB")
    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(5): _ = model.generate(**inputs, max_length=50)
    # Benchmark
    if device == "cuda": torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model.generate(**inputs, max_length=50)
            if device == "cuda": torch.cuda.synchronize()
    avg_time = (time.time() - start) / num_runs * 1000  # ms
    # Measure memory
    memory_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024) if device == "cuda" else 0
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    # Decode output
    caption = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    return {"avg_time_ms": avg_time, "memory_mb": memory_mb, "output_text": caption}
# Test image + baseline run
test_image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
print("\n=== Baseline: FP32 ===")
fp32_metrics = measure_inference_time(model_fp32, processor, test_image_url)
print(f"Inference time: {fp32_metrics['avg_time_ms']:.2f} ms"); print(f"GPU memory: {fp32_metrics['memory_mb']:.2f} MB")
print(f"Caption: {fp32_metrics['output_text']}")
