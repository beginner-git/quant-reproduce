"""Tests for common.eval.memory."""
import pytest
import torch

from common.eval.memory import measure_weight_memory, peak_gpu_memory
from common.models import load_hf_model


def test_measure_weight_memory_returns_dict_with_bytes(tiny_model_id):
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None)
    mem = measure_weight_memory(model)
    assert "weights_bytes" in mem
    assert "buffers_bytes" in mem
    assert mem["weights_bytes"] > 0
    assert mem["buffers_bytes"] >= 0


def test_measure_weight_memory_dtype_scaling(tiny_model_id):
    """fp32 权重字节数应是 fp16 的 ~2 倍。"""
    m32 = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None)
    m16 = load_hf_model(tiny_model_id, dtype=torch.float16, device_map=None)
    b32 = measure_weight_memory(m32)["weights_bytes"]
    b16 = measure_weight_memory(m16)["weights_bytes"]
    # 容差 ±5% 应付舍入差异
    assert 1.9 < (b32 / b16) < 2.1


@pytest.mark.require_cuda
def test_peak_gpu_memory_records_increase(device):
    with peak_gpu_memory(device) as p:
        x = torch.zeros(1024, 1024, dtype=torch.float32, device=device)
        del x
        torch.cuda.synchronize(device)
    # 1024 * 1024 * 4 bytes = 4 MB；应至少看到 1 MB 峰值
    assert p.bytes >= 1 * 2**20
