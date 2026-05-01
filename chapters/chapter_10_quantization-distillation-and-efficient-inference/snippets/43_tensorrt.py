import tensorrt as trt
def build_tensorrt_engine(onnx_path, engine_path="model.trt", precision="fp16",
                          max_batch_size=8, workspace_size_gb=4):
    """
