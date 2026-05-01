import torch.nn.functional as F

class FakeQuantize(torch.nn.Module):
    """Fake quantization module that simulates quantization during training while preserving full-precision gradients."""
    def __init__(self, num_bits=8, symmetric=True):
        super().__init__()
        self.num_bits, self.symmetric = num_bits, symmetric
        self.register_buffer("scale", torch.tensor(1.0)); self.register_buffer("zero_point", torch.tensor(0.0))
        self.qmin = -(2 ** (num_bits - 1)) if symmetric else 0
