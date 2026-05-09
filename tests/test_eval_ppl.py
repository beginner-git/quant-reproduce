"""Tests for common.eval.ppl."""
import math

import pytest
import torch

from common.eval.ppl import compute_ppl
from common.models import load_hf_model, load_tokenizer


@pytest.mark.require_cuda
def test_compute_ppl_finite_and_positive(tiny_model_id, device):
    tok = load_tokenizer(tiny_model_id)
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None).to(device)
    text = "The quick brown fox jumps over the lazy dog. " * 50
    tokens = tok(text, return_tensors="pt").input_ids[0]
    ppl = compute_ppl(model, tokens, seq_len=64, stride=64, device=device)
    assert math.isfinite(ppl)
    assert ppl > 0


@pytest.mark.require_cuda
def test_compute_ppl_deterministic(tiny_model_id, device):
    """同模型同输入两次结果应完全一致。"""
    tok = load_tokenizer(tiny_model_id)
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None).to(device)
    text = "Hello world. " * 100
    tokens = tok(text, return_tensors="pt").input_ids[0]
    a = compute_ppl(model, tokens, seq_len=64, stride=64, device=device)
    b = compute_ppl(model, tokens, seq_len=64, stride=64, device=device)
    assert a == b


@pytest.mark.require_cuda
def test_compute_ppl_short_input_raises(tiny_model_id, device):
    """tokens 不够一个完整窗口应明确报错而非返回 NaN。"""
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None).to(device)
    tokens = torch.zeros(10, dtype=torch.long)  # 远短于 seq_len=64
    with pytest.raises(ValueError, match="too short"):
        compute_ppl(model, tokens, seq_len=64, stride=64, device=device)
