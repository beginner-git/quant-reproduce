"""Tests for common.data."""
import torch

from common.data import load_c4_calibration, load_wikitext2_test
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


def test_load_c4_calibration_returns_list_of_long_tensors(tiny_model_id):
    tok = load_tokenizer(tiny_model_id)
    samples = load_c4_calibration(tok, n_samples=4, seq_len=128, seed=42)
    assert isinstance(samples, list)
    assert len(samples) == 4
    for s in samples:
        assert isinstance(s, torch.Tensor)
        assert s.dtype == torch.long
        assert s.shape == (128,)


def test_load_c4_calibration_seeded_deterministic(tiny_model_id):
    """同 seed 应产出完全一致的 token 序列。"""
    tok = load_tokenizer(tiny_model_id)
    a = load_c4_calibration(tok, n_samples=4, seq_len=128, seed=42)
    b = load_c4_calibration(tok, n_samples=4, seq_len=128, seed=42)
    for sa, sb in zip(a, b):
        assert torch.equal(sa, sb)


def test_load_c4_calibration_seed_changes_output(tiny_model_id):
    tok = load_tokenizer(tiny_model_id)
    a = load_c4_calibration(tok, n_samples=2, seq_len=128, seed=42)
    b = load_c4_calibration(tok, n_samples=2, seq_len=128, seed=7)
    # 至少其中一个样本不同
    assert not all(torch.equal(sa, sb) for sa, sb in zip(a, b))
