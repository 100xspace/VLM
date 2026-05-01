model.eval()
    torch.onnx.export(
        model, dummy_input, output_path,
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "sequence"}, "output": {0: "batch_size", 1: "sequence"}}
    )
    onnx_model = onnx.load(output_path); onnx.checker.check_model(onnx_model)
    print(f"Model exported to {output_path}")
    return output_path
def optimize_onnx_model(onnx_path, optimized_path="model_optimized.onnx"):
    """
