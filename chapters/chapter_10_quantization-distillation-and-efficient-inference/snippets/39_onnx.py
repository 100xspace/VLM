import time
import torch
import onnx
import onnxruntime as ort
def export_to_onnx(model, dummy_input, output_path="model.onnx"):
    """
