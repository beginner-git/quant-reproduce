"""Dataset loaders for PPL evaluation and PTQ calibration.

Locks the GPTQ paper protocol:
  - WikiText-2 test split: concat with '\\n\\n', tokenize as one long sequence.
  - C4 calibration: random non-overlapping seq_len windows, seeded. (Task 4)
"""
import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


def load_wikitext2_test(tokenizer: PreTrainedTokenizerBase) -> torch.LongTensor:
    """Load WikiText-2 raw test split, concatenate with '\\n\\n' (GPTQ protocol), return 1D LongTensor."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    encoded = tokenizer(text, return_tensors="pt")
    return encoded.input_ids[0]
