"""Smoke test for common.eval.zeroshot.

lm-eval-harness 跑全套 6 项即使 tiny model 也要数分钟，所以本测试只跑
piqa 一项 + limit=5 验证返回结构正确。
"""
import pytest
import torch

from common.eval.zeroshot import evaluate_zeroshot
from common.models import load_hf_model, load_tokenizer


@pytest.mark.require_cuda
@pytest.mark.slow
def test_evaluate_zeroshot_smoke_piqa(tiny_model_id, device):
    tok = load_tokenizer(tiny_model_id)
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None).to(device)

    results = evaluate_zeroshot(
        model, tok,
        tasks=["piqa"],
        limit=5,            # 仅 5 个样本，验证返回结构
        batch_size=1,
    )
    assert "piqa" in results
    assert 0.0 <= results["piqa"] <= 1.0
