"""Memory-footprint utilities.

`measure_weight_memory` reads the actual byte size of state_dict — for fake-quant
models this is still FP16 (which is the truth). Only models with packed-int
storage (e.g. autoawq's QuantLinear, auto-gptq's QuantLinear) report the
quantized size. Each method's README must declare which case applies.
"""
import contextlib
from dataclasses import dataclass

import torch
from torch import nn


def measure_weight_memory(model: nn.Module) -> dict:
    """Sum bytes of all parameters and buffers."""
    weights_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    buffers_bytes = sum(b.element_size() * b.numel() for b in model.buffers())
    return {"weights_bytes": weights_bytes, "buffers_bytes": buffers_bytes}


@dataclass
class _PeakHolder:
    bytes: int = 0


@contextlib.contextmanager
def peak_gpu_memory(device: str = "cuda"):
    """Context manager that captures peak GPU memory during the block.

    Usage:
        with peak_gpu_memory("cuda") as p:
            ...
        print(p.bytes)
    """
    if not torch.cuda.is_available():
        yield _PeakHolder(bytes=0)
        return
    torch.cuda.reset_peak_memory_stats(device)
    holder = _PeakHolder()
    try:
        yield holder
    finally:
        holder.bytes = torch.cuda.max_memory_allocated(device)
