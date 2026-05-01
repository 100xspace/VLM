"""Reusable quantization utilities for Chapter 10.

The functions here keep the math explicit so readers can inspect each step:
scale selection, clipping, rounding, dequantization, and error measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

QuantMode = Literal["symmetric", "asymmetric"]


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    num_bits: int
    mode: QuantMode
    axis: int | None = None


def _qrange(num_bits: int, mode: QuantMode) -> tuple[int, int]:
    if num_bits < 2:
        raise ValueError("num_bits must be >= 2")
    if mode == "symmetric":
        return -(2 ** (num_bits - 1) - 1), 2 ** (num_bits - 1) - 1
    if mode == "asymmetric":
        return 0, 2**num_bits - 1
    raise ValueError(f"Unknown quantization mode: {mode}")


def quantize_tensor(
    tensor: torch.Tensor,
    num_bits: int = 8,
    mode: QuantMode = "symmetric",
    axis: int | None = None,
    eps: float = 1e-8,
) -> QuantizedTensor:
    """Quantize a tensor with per-tensor or per-channel parameters."""
    if not tensor.is_floating_point():
        tensor = tensor.float()

    qmin, qmax = _qrange(num_bits, mode)
    if axis is None:
        reduce_dims: tuple[int, ...] | None = None
    else:
        axis = axis % tensor.ndim
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != axis)

    if mode == "symmetric":
        max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=True) if reduce_dims else tensor.abs().max()
        scale = torch.clamp(max_abs / qmax, min=eps)
        zero_point = torch.zeros_like(scale)
    else:
        min_val = tensor.amin(dim=reduce_dims, keepdim=True) if reduce_dims else tensor.min()
        max_val = tensor.amax(dim=reduce_dims, keepdim=True) if reduce_dims else tensor.max()
        scale = torch.clamp((max_val - min_val) / float(qmax - qmin), min=eps)
        zero_point = torch.round(qmin - min_val / scale).clamp(qmin, qmax)

    values = torch.round(tensor / scale + zero_point).clamp(qmin, qmax).to(torch.int8 if mode == "symmetric" else torch.uint8)
    return QuantizedTensor(values=values, scale=scale, zero_point=zero_point, num_bits=num_bits, mode=mode, axis=axis)


def dequantize_tensor(qtensor: QuantizedTensor) -> torch.Tensor:
    """Map quantized integer values back to approximate floating-point values."""
    return (qtensor.values.float() - qtensor.zero_point) * qtensor.scale


def quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    diff = (original.float() - reconstructed.float()).abs()
    denom = original.float().abs().mean().clamp_min(1e-8)
    return {
        "mae": float(diff.mean()),
        "max_abs_error": float(diff.max()),
        "relative_mae": float(diff.mean() / denom),
    }


def groupwise_quantize(tensor: torch.Tensor, group_size: int = 64, num_bits: int = 4) -> tuple[QuantizedTensor, int]:
    """Quantize the last dimension in fixed-size groups.

    Padding is added only for math convenience and returned so callers can trim
    the dequantized result back to the original width.
    """
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    width = tensor.shape[-1]
    pad = (group_size - width % group_size) % group_size
    if pad:
        tensor = torch.nn.functional.pad(tensor, (0, pad))
    grouped = tensor.reshape(*tensor.shape[:-1], -1, group_size)
    return quantize_tensor(grouped, num_bits=num_bits, mode="symmetric", axis=-1), pad


def dequantize_groupwise(qtensor: QuantizedTensor, pad: int = 0) -> torch.Tensor:
    restored = dequantize_tensor(qtensor).reshape(*qtensor.values.shape[:-2], -1)
    if pad:
        restored = restored[..., :-pad]
    return restored


def model_size_mb(model: torch.nn.Module) -> float:
    params = sum(param.numel() * param.element_size() for param in model.parameters())
    buffers = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return (params + buffers) / 1_000_000


if __name__ == "__main__":
    torch.manual_seed(7)
    weights = torch.randn(4, 128) * 0.7

    per_tensor = quantize_tensor(weights, num_bits=8, mode="symmetric")
    per_channel = quantize_tensor(weights, num_bits=8, mode="symmetric", axis=0)
    groupwise, pad = groupwise_quantize(weights, group_size=32, num_bits=4)

    print("Per-tensor INT8:", quantization_error(weights, dequantize_tensor(per_tensor)))
    print("Per-channel INT8:", quantization_error(weights, dequantize_tensor(per_channel)))
    print("Group-wise INT4:", quantization_error(weights, dequantize_groupwise(groupwise, pad)))
