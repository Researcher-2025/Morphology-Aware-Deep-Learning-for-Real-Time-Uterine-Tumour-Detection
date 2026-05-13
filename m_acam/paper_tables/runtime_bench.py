"""
Measure FPS and approximate GPU/CPU memory for a PyTorch module (inference only).
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn


def model_memory_estimate_mb(model: nn.Module, trainable_only: bool = False) -> float:
    """Host-side parameter (+ buffer) memory in MB (does not include activations)."""
    total = 0
    for t in model.parameters():
        if trainable_only and not t.requires_grad:
            continue
        total += t.numel() * t.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024.0**2)


def peak_inference_memory_mb_cuda(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int],
) -> float:
    """Peak CUDA memory during one forward pass (MB), including activations."""
    if device.type != "cuda":
        raise ValueError("CUDA device required for peak_inference_memory_mb_cuda")
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    x = torch.randn(input_shape, device=device, dtype=torch.float32)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024.0**2)


def benchmark_fps(
    model: nn.Module,
    device: torch.device,
    input_shape: Tuple[int, int, int, int],
    warmup: int = 10,
    repeats: int = 100,
) -> float:
    model.eval()
    x = torch.randn(input_shape, device=device, dtype=torch.float32)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    return repeats / max(t1 - t0, 1e-9)
