session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    input_np = dummy_input.detach().cpu().numpy()
    # Warmup
    for _ in range(10): _ = session.run(None, {input_name: input_np})
    # Benchmark
    start = time.time()
    for _ in range(num_runs): _ = session.run(None, {input_name: input_np})
    end = time.time()
    avg_ms = (end - start) / num_runs * 1000
    throughput = num_runs / (end - start)
    print(f"Average inference time: {avg_ms:.2f} ms")
    print(f"Throughput: {throughput:.2f} inferences/second")
    return avg_ms
# Example usage
# model = SimpleCNN()
# dummy_input = torch.randn(1, 3, 224, 224)
# onnx_path = export_to_onnx(model, dummy_input)
# opt_path = optimize_onnx_model(onnx_path)
# benchmark_onnx_runtime(opt_path, dummy_input)
