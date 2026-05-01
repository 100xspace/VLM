scales = torch.clamp(max_vals / qmax, min=1e-8)

    # Quantize and reshape back
    quantized = torch.clamp(torch.round(grouped / scales), qmin, qmax)\
                 .to(torch.int8)\
                 .reshape(out_features, num_groups * group_size)
    return quantized, scales

# Example: INT4 group-wise quantization
weight = torch.randn(4096, 4096) * 0.1
qt_group, scales_group = quantize_groupwise(weight, num_bits=4, group_size=128)
print(f"Weight shape: {weight.shape}")
print(f"Quantized shape: {qt_group.shape}")
print(f"Scales shape: {scales_group.shape}")
print(f"Effective compression: {weight.element_size() * 8 / 4:.1f}x")
