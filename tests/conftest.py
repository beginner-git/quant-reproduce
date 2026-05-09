"""Shared pytest fixtures for common/ tests."""
import pytest
import torch

# 用 sshleifer 的 tiny GPT-2 做单元测试（~1.5 MB，下载快）
TINY_MODEL_ID = "sshleifer/tiny-gpt2"


def pytest_collection_modifyitems(config, items):
    """没有 CUDA 时自动跳过被 require_cuda 标记的测试。"""
    if torch.cuda.is_available():
        return
    skip_cuda = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "require_cuda" in item.keywords:
            item.add_marker(skip_cuda)


@pytest.fixture(scope="session")
def tiny_model_id():
    return TINY_MODEL_ID


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
