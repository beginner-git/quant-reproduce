"""Perplexity computation following the GPTQ paper protocol.

Default stride == seq_len means non-overlapping windows. Each window's NLL is
multiplied by seq_len (matching the GPTQ/AWQ/BiLLM eval convention) and averaged
over n_windows * seq_len before exp(). This mirrors the standard reproducible
implementation used across the four-paper family.
"""
import torch
from torch import nn


@torch.no_grad()
def compute_ppl(
    model: nn.Module,
    tokens: torch.LongTensor,
    seq_len: int = 2048,
    stride: int = 2048,
    device: str = "cuda",
) -> float:
    """Compute perplexity via sliding window over `tokens`.

    Args:
        model: HF causal LM (must accept input_ids and labels in forward).
        tokens: 1D LongTensor of token ids (e.g. from `load_wikitext2_test`).
        seq_len: window size.
        stride: window stride (== seq_len for GPTQ paper protocol).
        device: device string for input_ids.

    Returns:
        Perplexity as float.

    Raises:
        ValueError: if `tokens` is shorter than one full window.
    """
    n_tokens = tokens.numel()
    if n_tokens < seq_len:
        raise ValueError(
            f"tokens too short: have {n_tokens}, need >= seq_len={seq_len}"
        )

    model.eval()
    nlls: list[torch.Tensor] = []

    for begin in range(0, n_tokens - seq_len + 1, stride):
        end = begin + seq_len
        input_ids = tokens[begin:end].unsqueeze(0).to(device)
        outputs = model(input_ids, labels=input_ids)
        # outputs.loss is mean cross-entropy over (seq_len-1) shifted predictions;
        # GPTQ convention: scale by seq_len, then average over n_windows * seq_len.
        nlls.append(outputs.loss.float() * seq_len)

    total_nll = torch.stack(nlls).sum() / (len(nlls) * seq_len)
    return float(torch.exp(total_nll))
