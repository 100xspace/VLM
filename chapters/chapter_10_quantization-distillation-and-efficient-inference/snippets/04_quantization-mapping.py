return scale * (quantized_tensor.float() - zero_point)

# Example usage
original_weights = torch.randn(1000, 1000) * 0.5  # Simulated model weights
quantized, scale, zp = quantize_tensor(original_weights, num_bits=8, symmetric=True)  # Quantize to INT8
print(f"Original dtype: {original_weights.dtype}, size: {original_weights.element_size() * original_weights.numel() / 1e6:.2f} MB")
print(f"Quantized dtype: {quantized.dtype}, size: {quantized.element_size() * quantized.numel() / 1e6:.2f} MB")
print(f"Compression ratio: {original_weights.element_size() / quantized.element_size():.1f}x")

# Dequantize and measure error
dequantized = dequantize_tensor(quantized, scale, zp)
mse = torch.mean((original_weights - dequantized) ** 2)
print(f"Quantization MSE: {mse:.6f}")
