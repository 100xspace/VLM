import torch
import torch.nn as nn
import torch.nn.functional as F
# Example: Fuse LayerNorm + Linear
class FusedLayerNormLinear(nn.Module):
    """Fused LayerNorm + Linear operation for efficiency"""
    def __init__(self, normalized_shape, out_features):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.linear = nn.Linear(normalized_shape, out_features)
    def forward(self, x):
        # Unfused version (2 kernel launches):
        # x = self.layer_norm(x)
        # x = self.linear(x)
        # Fused intent (still using PyTorch ops here; real fusion uses compiled kernels)
        x_norm = F.layer_norm(x, self.layer_norm.normalized_shape, self.layer_norm.weight, self.layer_norm.bias)
        return F.linear(x_norm, self.linear.weight, self.linear.bias)
