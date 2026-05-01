import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
# Demonstrate basic convolution operation
def demonstrate_convolution():
    """Show how different kernels detect different features"""

    # Create a simple input image
    input_image = torch.randn(1, 1, 28, 28)  # (batch, channels, height, width)

    # Define different types of kernels
    kernels = {
        'edge_horizontal': torch.tensor([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=torch.float32),
        'edge_vertical': torch.tensor([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], dtype=torch.float32),
        'blur': torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.float32) / 9.0,
        'sharpen': torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float32)
    }

    results = {}
    for name, kernel in kernels.items():
        # Reshape kernel to match Conv2d expectations: (out_channels, in_channels, height, width)
        kernel_reshaped = kernel.view(1, 1, 3, 3)

        # Apply convolution
        output = F.conv2d(input_image, kernel_reshaped, padding=1)
        results[name] = output
    return input_image, results

# Example usage
input_img, conv_results = demonstrate_convolution()
print(f"Input shape: {input_img.shape}")
for name, output in conv_results.items():
    print(f"{name} output shape: {output.shape}")
