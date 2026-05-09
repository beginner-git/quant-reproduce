"""Tests for common.models."""
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from common.models import load_hf_model, load_tokenizer


def test_load_tokenizer_returns_tokenizer(tiny_model_id):
    tok = load_tokenizer(tiny_model_id)
    assert isinstance(tok, PreTrainedTokenizerBase)
    # 应该已设置 pad_token（fallback 到 eos_token）
    assert tok.pad_token is not None


def test_load_hf_model_returns_model(tiny_model_id):
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None)
    assert isinstance(model, PreTrainedModel)


def test_load_hf_model_respects_dtype(tiny_model_id):
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None)
    # 取一个权重看 dtype
    first_param = next(model.parameters())
    assert first_param.dtype == torch.float32
