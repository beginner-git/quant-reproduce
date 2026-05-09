"""Tests for common.data."""
import torch

from common.data import load_wikitext2_test
from common.models import load_tokenizer


def test_load_wikitext2_returns_1d_long_tensor(tiny_model_id):
    tok = load_tokenizer(tiny_model_id)
    tokens = load_wikitext2_test(tok)
    assert isinstance(tokens, torch.Tensor)
    assert tokens.dtype == torch.long
    assert tokens.dim() == 1
    # WikiText-2 test split tokenized 后应该至少有几万个 token
    assert tokens.numel() > 10_000


def test_load_wikitext2_deterministic(tiny_model_id):
    """同一 tokenizer 应产生完全一致的 token 序列。"""
    tok = load_tokenizer(tiny_model_id)
    a = load_wikitext2_test(tok)
    b = load_wikitext2_test(tok)
    assert torch.equal(a, b)
