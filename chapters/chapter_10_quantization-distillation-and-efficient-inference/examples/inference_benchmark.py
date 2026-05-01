"""Small benchmarking helpers for Chapter 10 inference experiments."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    p50_ms: float
    p95_ms: float
    mean_ms: float
    throughput_per_second: float
    device: str


def benchmark_callable(fn, warmup: int = 5, runs: int = 30, device: str = "cpu") -> BenchmarkResult:
    """Measure latency for a callable with optional CUDA synchronization."""
    if runs <= 0:
        raise ValueError("runs must be positive")
    synchronize = device.startswith("cuda") and torch.cuda.is_available()

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if synchronize:
            torch.cuda.synchronize()

        latencies = []
        for _ in range(runs):
            start = time.perf_counter()
            fn()
            if synchronize:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

    quantiles = statistics.quantiles(latencies, n=20, method="inclusive")
    mean_ms = statistics.fmean(latencies)
    return BenchmarkResult(
        p50_ms=statistics.median(latencies),
        p95_ms=quantiles[18],
        mean_ms=mean_ms,
        throughput_per_second=1000.0 / mean_ms,
        device=device,
    )


def demo_model(device: str) -> tuple[torch.nn.Module, torch.Tensor]:
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((8, 8)),
        torch.nn.Flatten(),
        torch.nn.Linear(32 * 8 * 8, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
    ).to(device)
    model.eval()
    inputs = torch.randn(8, 3, 224, 224, device=device)
    return model, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a small vision encoder.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    model, inputs = demo_model(args.device)
    result = benchmark_callable(lambda: model(inputs), runs=args.runs, device=args.device)
    print(result)

    if args.device == "cpu":
        quantized = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        q_result = benchmark_callable(lambda: quantized(inputs), runs=args.runs, device=args.device)
        print("dynamic_int8", q_result)


if __name__ == "__main__":
    main()
