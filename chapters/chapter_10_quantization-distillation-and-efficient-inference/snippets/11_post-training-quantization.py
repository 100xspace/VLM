import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from torch.quantization import quantize_dynamic, quantize_qat, get_default_qconfig
def apply_ptq_dynamic(model, dtype=torch.qint8):
    """
