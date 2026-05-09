"""Dataset loaders for PPL evaluation and PTQ calibration.

Locks the GPTQ paper protocol:
  - WikiText-2 test split: concat with '\\n\\n', tokenize as one long sequence.
  - C4 calibration: random non-overlapping seq_len windows, seeded.
"""

import random

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_wikitext2_test(tokenizer: PreTrainedTokenizerBase) -> torch.LongTensor:
    """Load WikiText-2 raw test split, concatenate with '\\n\\n' (GPTQ protocol), return 1D LongTensor."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encoded = tokenizer(text, return_tensors="pt")
    return encoded.input_ids[0]


def load_c4_calibration(
    tokenizer: PreTrainedTokenizerBase,
    n_samples: int = 128,
    seq_len: int = 2048,
    seed: int = 42,
) -> list[torch.LongTensor]:
    """Sample n_samples random seq_len-token windows from C4 validation split.

    GPTQ-style: skip docs shorter than seq_len; for each accepted doc, take a random offset.
    Streaming + seeded shuffle so we don't load all of C4 into memory.
    """
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
    ).shuffle(seed=seed, buffer_size=10_000)

    rng = random.Random(seed)
    samples: list[torch.LongTensor] = []

    for example in ds:
        if len(samples) >= n_samples:
            break
        tokens = tokenizer(example["text"], return_tensors="pt").input_ids[0]
        if tokens.shape[0] < seq_len:
            continue
        start = rng.randint(0, tokens.shape[0] - seq_len)
        samples.append(tokens[start : start + seq_len])

    if len(samples) < n_samples:
        raise RuntimeError(
            f"Only collected {len(samples)} / {n_samples} calibration samples; "
            f"increase buffer or check C4 streaming connectivity."
        )
    return samples
