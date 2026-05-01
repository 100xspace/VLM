import onnxoptimizer
    passes = [
        "eliminate_deadend","eliminate_identity","eliminate_nop_dropout","eliminate_nop_monotone_argmax","eliminate_nop_pad",
        "extract_constant_to_initializer","eliminate_unused_initializer","fuse_add_bias_into_conv","fuse_bn_into_conv",       "fuse_consecutive_concats","fuse_consecutive_reduce_unsqueeze","fuse_consecutive_squeezes","fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm","fuse_pad_into_conv","fuse_transpose_into_gemm",
    ]
    model = onnx.load(onnx_path)
    onnx.save(onnxoptimizer.optimize(model, passes), optimized_path)
    print(f"Optimized model saved to {optimized_path}")
    return optimized_path
def benchmark_onnx_runtime(onnx_path, dummy_input, num_runs=100):
    """
