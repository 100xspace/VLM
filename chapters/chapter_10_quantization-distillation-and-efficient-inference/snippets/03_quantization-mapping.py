quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)\
# INT8 uses int8 storage; lower bit-widths (e.g., INT4) are packed but represented here using int32 for simplicity
                         .to(torch.int8 if num_bits == 8 else torch.int32)
    else:
        # Asymmetric quantization: range is [min, max]
        min_val, max_val = torch.min(tensor), torch.max(tensor)
        qmax, qmin = 2 ** num_bits - 1, 0
        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - torch.round(min_val / scale)
        quantized = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)\
                         .to(torch.uint8 if num_bits == 8 else torch.int32)
    return quantized, scale, zero_point

def dequantize_tensor(quantized_tensor, scale, zero_point):
    """
