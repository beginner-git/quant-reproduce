"""Unified HF model & tokenizer loading.

Single entry point used by all four method subdirs (GPTQ/AWQ/BiLLM/KIVI) so that
each method runs against the same model commit and same dtype/device conventions.
"""
import os

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def _resolve_cache_dir(cache_dir: str | None) -> str:
    if cache_dir is not None:
        return cache_dir
    return os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")


def load_hf_model(
    model_id: str,
    dtype: torch.dtype = torch.float16,
    device_map: str | None = "auto",
    cache_dir: str | None = None,
) -> PreTrainedModel:
    """Load a causal-LM HF model with locked dtype / device_map / cache_dir conventions."""
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        cache_dir=_resolve_cache_dir(cache_dir),
        trust_remote_code=False,
    )


def load_tokenizer(model_id: str, cache_dir: str | None = None) -> PreTrainedTokenizerBase:
    """Load tokenizer; ensures pad_token is set (falls back to eos_token)."""
    tok = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=_resolve_cache_dir(cache_dir),
        use_fast=True,
        trust_remote_code=False,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
