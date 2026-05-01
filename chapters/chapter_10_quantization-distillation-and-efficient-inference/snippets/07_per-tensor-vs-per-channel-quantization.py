scales = torch.clamp(max_vals / qmax, min=1e-8)
    # Quantize
    quantized = torch.clamp(torch.round(weight_tensor / scales), qmin, qmax).to(torch.int8)
    return quantized, scales

# Example: Quantize a linear layer
linear_weight = torch.randn(4096, 4096) * 0.1  # [out_features, in_features]
# Per-tensor quantization
qt_tensor, scale_tensor, _ = quantize_tensor(linear_weight, num_bits=8)
# Per-channel quantization
qt_channel, scales_channel = quantize_per_channel(linear_weight, num_bits=8, dim=0)
print(f"Per-tensor scale: {scale_tensor:.6f}")
print(f"Per-channel scales shape: {scales_channel.shape}")
print(f"Per-channel scale range: [{scales_channel.min():.6f}, {scales_channel.max():.6f}]")
