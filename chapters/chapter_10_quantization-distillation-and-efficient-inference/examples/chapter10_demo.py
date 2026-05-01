"""Offline Chapter 10 demo covering quantization, distillation, and latency."""

from __future__ import annotations

import torch

from distillation_loss import vlm_distillation_loss
from inference_benchmark import benchmark_callable, demo_model
from quantization_tools import dequantize_tensor, quantization_error, quantize_tensor


def main() -> None:
    torch.manual_seed(10)

    weights = torch.randn(16, 64)
    qtensor = quantize_tensor(weights, num_bits=8, mode="symmetric", axis=0)
    restored = dequantize_tensor(qtensor)
    print("quantization", quantization_error(weights, restored))

    student_logits = torch.randn(4, 20, requires_grad=True)
    teacher_logits = student_logits.detach() + 0.4 * torch.randn(4, 20)
    labels = torch.randint(0, 20, (4,))
    loss, metrics = vlm_distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    print("distillation", metrics)

    model, inputs = demo_model("cpu")
    result = benchmark_callable(lambda: model(inputs), runs=10, device="cpu")
    print("latency", result)


if __name__ == "__main__":
    main()
