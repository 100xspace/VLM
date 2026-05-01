TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("ERROR: Failed to parse ONNX model")
            for i in range(parser.num_errors): print(parser.get_error(i))
            return None
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size_gb * (1 << 30)  # GB -> bytes
    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16); print("FP16 mode enabled")
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8); print("INT8 mode enabled")  # INT8 needs a calibrator
    # Set dynamic shapes (example profile)
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    profile.set_shape(inp.name, min=(1, 3, 224, 224), opt=(4, 3, 224, 224), max=(max_batch_size, 3, 224, 224))
    config.add_optimization_profile(profile)
    # Build engine
    print("Building TensorRT engine...")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("ERROR: Failed to build TensorRT engine")
        return None
    with open(engine_path, "wb") as f: f.write(engine.serialize())
    print(f"TensorRT engine saved to {engine_path}")
    return engine_path
class TensorRTInference:
    """Wrapper for TensorRT inference (expects CUDA buffer utilities available)"""
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate buffers (requires: pycuda.driver as cuda, numpy as np)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            (self.inputs if self.engine.binding_is_input(binding) else self.outputs).append(
                {"host": host_mem, "device": device_mem}
            )
    def infer(self, input_data):
        """Run inference"""
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod(self.inputs[0]["device"], self.inputs[0]["host"])
        self.context.execute_v2(bindings=self.bindings)
        cuda.memcpy_dtoh(self.outputs[0]["host"], self.outputs[0]["device"])
        return self.outputs[0]["host"]
# Example usage
# engine_path = build_tensorrt_engine("model.onnx", precision="fp16")
# trt_runner = TensorRTInference(engine_path)
# output = trt_runner.infer(input_data)
