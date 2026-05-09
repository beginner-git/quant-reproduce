# Bootstrap + GPTQ Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the project skeleton with shared `common/` infrastructure, reproduce GPTQ on LLaMA-2-7B with W4-g128 to within ±0.3 PPL of the published number, and write the Phase 2 source-code study notes for GPTQ.

**Architecture:** 顶层 `common/` 是纯 Python 包，承担 PPL/zero-shot/memory 评测、WikiText-2/C4 数据加载、HF 模型加载。`GPTQ/` 子目录有独立 conda env (`quant-gptq`)，调 `auto-gptq` 包做量化，调用 `common/eval` 出统一格式数字。`docs/reports/gptq.md` 是 Phase 2 源码研读的五节笔记。

**Tech Stack:** Python 3.10+，PyTorch 2.x，transformers，datasets，lm-eval，auto-gptq ≥ 0.7，pytest。Conda 做 env 隔离。Windows 11 (PowerShell) 本地 + Linux 实验室服务器双环境。

**Spec:** `docs/superpowers/specs/2026-05-09-quant-reproduce-design.md`

---

## File Structure

本计划创建/修改的文件（按目录分组）：

```
F:\CODE\Quant\reproduce\
├── pyproject.toml                       # 创建：定义 common 包 + dev tooling
├── README.md                            # 创建：项目入口 + 方法表 + 链接 GPTQ
│
├── common/
│   ├── __init__.py                      # 创建（空）
│   ├── data.py                          # 创建：load_wikitext2_test, load_c4_calibration
│   ├── models.py                        # 创建：load_hf_model, load_tokenizer
│   └── eval/
│       ├── __init__.py                  # 创建（空）
│       ├── ppl.py                       # 创建：compute_ppl
│       ├── memory.py                    # 创建：measure_weight_memory + peak_gpu_memory
│       └── zeroshot.py                  # 创建：evaluate_zeroshot
│
├── tests/
│   ├── conftest.py                      # 创建：tiny_model_id fixture + cuda check
│   ├── test_models.py                   # 创建
│   ├── test_data.py                     # 创建
│   ├── test_eval_ppl.py                 # 创建
│   ├── test_eval_memory.py              # 创建
│   └── test_eval_zeroshot.py            # 创建（smoke only）
│
├── scripts/
│   ├── env_local.ps1                    # 创建：本地 PowerShell 一键建 env
│   ├── env_lab.sh                       # 创建：Linux lab bash 一键建 env
│   └── run_phase1_method.sh             # 创建：参数化跑某方法的 repro.py
│
├── GPTQ/
│   ├── README.md                        # 创建：用法 + 数字 + troubleshooting
│   ├── env.yml                          # 创建：conda env 完整定义
│   ├── requirements.txt                 # 创建：pip 依赖锁定（与 env.yml 同步）
│   ├── repro.py                         # 创建：CLI 入口
│   └── results/                         # 创建（空），canonical 跑后填充
│       ├── results_w4g128.json          # 由 Task 13 填入
│       ├── meta_w4g128.json             # 由 Task 13 填入
│       └── ppl_w4g128_raw.csv           # 由 Task 13 填入（每片 NLL）
│
└── docs/
    ├── reports/
    │   └── gptq.md                      # 创建（Task 14）：Phase 2 五节笔记
    └── results/
        └── summary.md                   # 创建（Task 15）：仅 GPTQ 一行
```

**已存在不动**：`.gitignore`，`docs/superpowers/specs/2026-05-09-quant-reproduce-design.md`。

**单文件职责**：
- `common/data.py` — 仅 dataset 加载，无算法。
- `common/eval/*.py` — 各文件一种评测维度；不交叉。
- `common/models.py` — HF 加载唯一入口；不放算法逻辑。
- `GPTQ/repro.py` — 端到端 CLI 入口；调 `common/` + `auto-gptq` 完成 calibration → quantize → eval → 落盘。
- `tests/test_*.py` — 一文件对应 `common/` 一模块。

---

## Task 1: Bootstrap 项目骨架（pyproject.toml + 顶层 README + 空的 common 包）

**Files:**
- Create: `F:\CODE\Quant\reproduce\pyproject.toml`
- Create: `F:\CODE\Quant\reproduce\README.md`
- Create: `F:\CODE\Quant\reproduce\common\__init__.py`
- Create: `F:\CODE\Quant\reproduce\common\eval\__init__.py`
- Create: `F:\CODE\Quant\reproduce\tests\conftest.py`

- [ ] **Step 1: 创建 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "quant-reproduce-common"
version = "0.1.0"
description = "Shared eval / data / model-loading layer for the LLM quantization reproduction portfolio."
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.27",
    "lm-eval>=0.4.2",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-xdist>=3.5",
    "ruff>=0.4",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["common*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"

[tool.ruff]
line-length = 110
target-version = "py310"
```

- [ ] **Step 2: 创建顶层 `README.md`（占位，含方法表）**

```markdown
# LLM 量化论文复现

> 申请研究生作品集中的核心工程作品。三阶段：复现 → 源码研读 → 自写统一 pipeline。
> 设计文档：[`docs/superpowers/specs/2026-05-09-quant-reproduce-design.md`](docs/superpowers/specs/2026-05-09-quant-reproduce-design.md)

## 方法

| 方法 | 论文 | 状态 | 链接 |
|------|------|------|------|
| **GPTQ**  | Frantar et al., ICLR 2023  | 进行中 | [GPTQ/](GPTQ/) |
| **AWQ**   | Lin et al., MLSys 2024     | 待开始 | — |
| **BiLLM** | Huang et al., ICML 2024    | 待开始 | — |
| **KIVI**  | Liu et al., ICML 2024      | 待开始 | — |

## 数字汇总

见 [`docs/results/summary.md`](docs/results/summary.md)。

## 环境

每方法独立 conda env，避免依赖冲突。详见各方法子目录 README 与 `scripts/`。
```

- [ ] **Step 3: 创建空的 `common/__init__.py` 和 `common/eval/__init__.py`**

两个文件都是空（0 字节）。Python 用它们识别 `common` 和 `common.eval` 为包。

- [ ] **Step 4: 创建 `tests/conftest.py` 含 tiny model fixture 和 cuda 跳过装饰器**

```python
"""Shared pytest fixtures for common/ tests."""
import os
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
```

- [ ] **Step 5: 验证 pip install 可工作**

执行：
```powershell
cd F:\CODE\Quant\reproduce
pip install -e .[dev]
```

预期：成功安装；之后 `python -c "import common"` 应静默成功。

- [ ] **Step 6: 验证 pytest 可发现 0 个测试**

执行：
```powershell
pytest --collect-only
```

预期：输出 "collected 0 items"，没有 ERROR。

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml README.md common/__init__.py common/eval/__init__.py tests/conftest.py
git commit -m "feat: bootstrap project skeleton (pyproject + common pkg + pytest)"
```

---

## Task 2: 实现 `common.models`（TDD）

**Files:**
- Create: `F:\CODE\Quant\reproduce\tests\test_models.py`
- Create: `F:\CODE\Quant\reproduce\common\models.py`

- [ ] **Step 1: 写失败的测试 `tests/test_models.py`**

```python
"""Tests for common.models."""
import pytest
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
```

- [ ] **Step 2: 跑测试确认失败**

执行：
```powershell
pytest tests/test_models.py -v
```

预期：所有三个测试 FAIL（`ModuleNotFoundError: No module named 'common.models'`）。

- [ ] **Step 3: 实现 `common/models.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

执行：
```powershell
pytest tests/test_models.py -v
```

预期：3 PASSED。第一次跑会下载 tiny-gpt2（~5 MB，几秒）。

- [ ] **Step 5: Commit**

```powershell
git add common/models.py tests/test_models.py
git commit -m "feat(common): add HF model/tokenizer loader with unified cache_dir & dtype"
```

---

## Task 3: 实现 `common.data.load_wikitext2_test`（TDD）

**Files:**
- Create: `F:\CODE\Quant\reproduce\tests\test_data.py`
- Create: `F:\CODE\Quant\reproduce\common\data.py`

- [ ] **Step 1: 写失败的测试**

```python
"""Tests for common.data."""
import pytest
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
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
pytest tests/test_data.py -v
```

预期：FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `common/data.py`（先只放 wikitext2）**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

```powershell
pytest tests/test_data.py -v
```

预期：2 PASSED。第一次会下载 wikitext-2-raw-v1（~5 MB）。

- [ ] **Step 5: Commit**

```powershell
git add common/data.py tests/test_data.py
git commit -m "feat(common): add WikiText-2 test loader (GPTQ protocol)"
```

---

## Task 4: 实现 `common.data.load_c4_calibration`（TDD）

**Files:**
- Modify: `F:\CODE\Quant\reproduce\common\data.py`
- Modify: `F:\CODE\Quant\reproduce\tests\test_data.py`

- [ ] **Step 1: 追加失败的测试到 `tests/test_data.py`**

在文件末尾追加：
```python
from common.data import load_c4_calibration


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
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
pytest tests/test_data.py::test_load_c4_calibration_returns_list_of_long_tensors -v
```

预期：FAIL（`ImportError: cannot import name 'load_c4_calibration'`）。

- [ ] **Step 3: 在 `common/data.py` 末尾追加实现**

```python


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
```

- [ ] **Step 4: 跑全部 data 测试确认通过**

```powershell
pytest tests/test_data.py -v
```

预期：5 PASSED（前 2 + 新增 3）。第一次会下载几个 C4 example（~1 MB）。

- [ ] **Step 5: Commit**

```powershell
git add common/data.py tests/test_data.py
git commit -m "feat(common): add C4 calibration loader (seeded, GPTQ-style)"
```

---

## Task 5: 实现 `common.eval.ppl.compute_ppl`（TDD）

**Files:**
- Create: `F:\CODE\Quant\reproduce\tests\test_eval_ppl.py`
- Create: `F:\CODE\Quant\reproduce\common\eval\ppl.py`

- [ ] **Step 1: 写失败的测试**

```python
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
    tok = load_tokenizer(tiny_model_id)
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None).to(device)
    tokens = torch.zeros(10, dtype=torch.long)  # 远短于 seq_len=64
    with pytest.raises(ValueError, match="too short"):
        compute_ppl(model, tokens, seq_len=64, stride=64, device=device)
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
pytest tests/test_eval_ppl.py -v
```

预期：3 FAIL（`ModuleNotFoundError`）。

- [ ] **Step 3: 实现 `common/eval/ppl.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

```powershell
pytest tests/test_eval_ppl.py -v
```

预期：3 PASSED（如果有 CUDA），否则 SKIPPED。

- [ ] **Step 5: Commit**

```powershell
git add common/eval/ppl.py tests/test_eval_ppl.py
git commit -m "feat(common): add compute_ppl (GPTQ paper protocol, sliding-window NLL)"
```

---

## Task 6: 实现 `common.eval.memory`（TDD）

**Files:**
- Create: `F:\CODE\Quant\reproduce\tests\test_eval_memory.py`
- Create: `F:\CODE\Quant\reproduce\common\eval\memory.py`

> KV cache profiler (`measure_kv_cache_bytes`) 在本计划中暂不实现 — 它会在 Plan D (KIVI) 中添加，避免现在过早设计接口。

- [ ] **Step 1: 写失败的测试**

```python
"""Tests for common.eval.memory."""
import pytest
import torch

from common.eval.memory import measure_weight_memory, peak_gpu_memory
from common.models import load_hf_model


def test_measure_weight_memory_returns_dict_with_bytes(tiny_model_id):
    model = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None)
    mem = measure_weight_memory(model)
    assert "weights_bytes" in mem
    assert "buffers_bytes" in mem
    assert mem["weights_bytes"] > 0
    assert mem["buffers_bytes"] >= 0


def test_measure_weight_memory_dtype_scaling(tiny_model_id):
    """fp32 权重字节数应是 fp16 的 ~2 倍。"""
    m32 = load_hf_model(tiny_model_id, dtype=torch.float32, device_map=None)
    m16 = load_hf_model(tiny_model_id, dtype=torch.float16, device_map=None)
    b32 = measure_weight_memory(m32)["weights_bytes"]
    b16 = measure_weight_memory(m16)["weights_bytes"]
    # 容差 ±5% 应付舍入差异
    assert 1.9 < (b32 / b16) < 2.1


@pytest.mark.require_cuda
def test_peak_gpu_memory_records_increase(device):
    with peak_gpu_memory(device) as p:
        x = torch.zeros(1024, 1024, dtype=torch.float32, device=device)
        del x
        torch.cuda.synchronize(device)
    # 1024 * 1024 * 4 bytes = 4 MB；应至少看到 1 MB 峰值
    assert p.bytes >= 1 * 2**20
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
pytest tests/test_eval_memory.py -v
```

预期：3 FAIL（ModuleNotFoundError）。

- [ ] **Step 3: 实现 `common/eval/memory.py`**

```python
"""Memory-footprint utilities.

`measure_weight_memory` reads the actual byte size of state_dict — for fake-quant
models this is still FP16 (which is the truth). Only models with packed-int
storage (e.g. autoawq's QuantLinear, auto-gptq's QuantLinear) report the
quantized size. Each method's README must declare which case applies.
"""
import contextlib
from dataclasses import dataclass

import torch
from torch import nn


def measure_weight_memory(model: nn.Module) -> dict:
    """Sum bytes of all parameters and buffers."""
    weights_bytes = sum(p.element_size() * p.numel() for p in model.parameters())
    buffers_bytes = sum(b.element_size() * b.numel() for b in model.buffers())
    return {"weights_bytes": weights_bytes, "buffers_bytes": buffers_bytes}


@dataclass
class _PeakHolder:
    bytes: int = 0


@contextlib.contextmanager
def peak_gpu_memory(device: str = "cuda"):
    """Context manager that captures peak GPU memory during the block.

    Usage:
        with peak_gpu_memory("cuda") as p:
            ...
        print(p.bytes)
    """
    if not torch.cuda.is_available():
        yield _PeakHolder(bytes=0)
        return
    torch.cuda.reset_peak_memory_stats(device)
    holder = _PeakHolder()
    try:
        yield holder
    finally:
        holder.bytes = torch.cuda.max_memory_allocated(device)
```

- [ ] **Step 4: 跑测试确认通过**

```powershell
pytest tests/test_eval_memory.py -v
```

预期：3 PASSED（CUDA 可用）。

- [ ] **Step 5: Commit**

```powershell
git add common/eval/memory.py tests/test_eval_memory.py
git commit -m "feat(common): add weight memory profiler + peak_gpu_memory context"
```

---

## Task 7: 实现 `common.eval.zeroshot`（仅 smoke 测试，因 lm-eval 太慢）

**Files:**
- Create: `F:\CODE\Quant\reproduce\tests\test_eval_zeroshot.py`
- Create: `F:\CODE\Quant\reproduce\common\eval\zeroshot.py`

- [ ] **Step 1: 写 smoke 测试**

```python
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
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
pytest tests/test_eval_zeroshot.py -v
```

预期：FAIL（ModuleNotFoundError）。

- [ ] **Step 3: 实现 `common/eval/zeroshot.py`**

```python
"""lm-eval-harness wrapper.

Standardises the task list (matching the AWQ paper's reported subset) and the
return shape (a flat dict {task: accuracy}) so the four method subdirs all
emit the same JSON schema.
"""
from typing import Sequence

from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


DEFAULT_TASKS: list[str] = [
    "piqa",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "winogrande",
    "openbookqa",
]


def evaluate_zeroshot(
    model,
    tokenizer,
    tasks: Sequence[str] = DEFAULT_TASKS,
    num_fewshot: int = 0,
    batch_size: int = 1,
    limit: int | None = None,
) -> dict[str, float]:
    """Run lm-eval-harness 0-shot suite, return {task: accuracy}.

    Args:
        model: HF model already on target device.
        tokenizer: matching HF tokenizer.
        tasks: list of lm-eval task names.
        num_fewshot: 0 for zero-shot.
        batch_size: harness batch size.
        limit: cap samples per task (use small int for smoke tests).

    Returns:
        {task_name: accuracy_float}. Uses the "acc,none" metric reported by lm-eval.
    """
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    out = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
    )
    return {task: out["results"][task]["acc,none"] for task in tasks}
```

- [ ] **Step 4: 给 pyproject.toml 加 slow 标记定义**

修改 `pyproject.toml` 中的 `[tool.pytest.ini_options]` 段为：
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short"
markers = [
    "require_cuda: skip when CUDA not available",
    "slow: lm-eval-harness or canonical-model tests; run explicitly with -m slow",
]
```

- [ ] **Step 5: 跑 smoke 测试确认通过**

```powershell
pytest tests/test_eval_zeroshot.py -v -m slow
```

预期：1 PASSED（约 30s–2min，含 lm-eval 数据下载）。

- [ ] **Step 6: Commit**

```powershell
git add common/eval/zeroshot.py tests/test_eval_zeroshot.py pyproject.toml
git commit -m "feat(common): add lm-eval-harness wrapper + slow/cuda pytest markers"
```

---

## Task 8: 创建 GPTQ 子目录骨架（README + env.yml + requirements.txt + 空 repro.py）

**Files:**
- Create: `F:\CODE\Quant\reproduce\GPTQ\README.md`
- Create: `F:\CODE\Quant\reproduce\GPTQ\env.yml`
- Create: `F:\CODE\Quant\reproduce\GPTQ\requirements.txt`
- Create: `F:\CODE\Quant\reproduce\GPTQ\repro.py`（占位 stub）
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\.gitkeep`

- [ ] **Step 1: 创建 `GPTQ/README.md`（数字部分先放占位，Task 13 后填实数）**

```markdown
# GPTQ — Phase 1 复现

> Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023
> 论文：<https://arxiv.org/abs/2210.17323>
> 用社区 fork [`auto-gptq`](https://github.com/AutoGPTQ/AutoGPTQ) 跑（原版主要支持 OPT，社区 fork 是 LLaMA-2 的事实标准）。

## 跑法

### 一次性建 env（每台机器跑一次）

PowerShell（Windows 本地）：
```powershell
conda env create -f GPTQ\env.yml
conda activate quant-gptq
pip install -e ..   # 装顶层 common
```

bash（Linux 实验室）：
```bash
conda env create -f GPTQ/env.yml
conda activate quant-gptq
pip install -e ..
```

### 烟雾测试（本地 12GB 5–10 min）

```bash
python repro.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --config w4g128 --calib-samples 32 --seq-len 1024 --eval ppl --smoke
```

### Canonical 跑（lab 96GB GPU）

```bash
python repro.py --model meta-llama/Llama-2-7b-hf --config w4g128 --eval ppl,zeroshot,memory --out results/
```

输出：
- `results/results_w4g128.json` — PPL / zero-shot / memory 数字
- `results/meta_w4g128.json` — 复现元数据（pkg 版本、HF SHA、时间、GPU）
- `results/ppl_w4g128_raw.csv` —（可选）每片 NLL，用于复盘异常

## 实测 vs 论文

| Config | Model | Metric | 实测 | 论文 anchor | 容差判定 |
|--------|-------|--------|------|-------------|---------|
| w4g128 | LLaMA-2-7B | WT2 PPL | _TBD_ | ≈ 5.69 | ±0.3 |
| w4g128 | LLaMA-2-7B | piqa    | _TBD_ | _TBD_       | ±0.02 |
| w4g128 | LLaMA-2-7B | weights MB | _TBD_ | ≈ 3.7 GB | — |

> 数字将在 `results/results_w4g128.json` 落地后填入。

## Troubleshooting（数字打不到目标的排查顺序）

1. 模型 ID 与 commit SHA — `meta_w4g128.json` 里 `model` 字段对照 HF 模型卡。
2. Calibration 切片 — seed=42、n_samples=128、seq_len=2048 全部固定？
3. auto-gptq 版本 — 论文出版前后 auto-gptq 自己也改过实现；锁定 ≥ 0.7 < 1.0。
4. 把 auto-gptq 自己 examples 目录的 quantize_with_alpaca.py 跑一遍数字对照。
5. 仍不通：在本 README 末尾"实测异常记录"小节如实写实测 + 已排查步骤。**不要刷参数。**

## 算法摘要

GPTQ 用近似的二阶信息（Hessian）做 layer-wise 量化：每层逐列量化，每列量化后用其余列的 Hessian 信息更新剩余列以补偿误差。Cholesky 分解避免显式求逆，关键技巧是按 diag(H) 降序选列以减少误差累积。

详细源码研读：[`docs/reports/gptq.md`](../docs/reports/gptq.md)（Phase 2 输出）
```

- [ ] **Step 2: 创建 `GPTQ/env.yml`**

```yaml
name: quant-gptq
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch>=2.1
  - pytorch-cuda=12.1
  - pip:
    - -r requirements.txt
```

- [ ] **Step 3: 创建 `GPTQ/requirements.txt`**

```
auto-gptq>=0.7,<1.0
transformers>=4.40
accelerate>=0.27
datasets>=2.18
lm-eval>=0.4.2
optimum>=1.20
```

- [ ] **Step 4: 创建 `GPTQ/repro.py` 占位（仅 docstring + main stub，Task 9–10 实现）**

```python
"""GPTQ Phase 1 reproduction entry point.

Run `python repro.py --help` to see CLI options. Full implementation lives in
Tasks 9–10 of `docs/superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md`.
"""
import sys


def main():
    raise NotImplementedError("Stub — see Task 9 of the bootstrap plan.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 创建 `GPTQ/results/.gitkeep`**

空文件（让空目录进 git）。

- [ ] **Step 6: Commit**

```powershell
git add GPTQ/
git commit -m "scaffold(gptq): create subdir with env.yml, requirements, README template"
```

---

## Task 9: 实现 `GPTQ/repro.py` 的 argparse + presets（TDD）

**Files:**
- Create: `F:\CODE\Quant\reproduce\GPTQ\test_repro_cli.py`
- Modify: `F:\CODE\Quant\reproduce\GPTQ\repro.py`

> 仅这一部分用 TDD（CLI 解析逻辑可单元测）。Task 10 的端到端 main flow 走 smoke run 验证。

- [ ] **Step 1: 写失败的测试 `GPTQ/test_repro_cli.py`**

```python
"""CLI argument parsing tests for GPTQ/repro.py.

Tests are colocated with repro.py rather than under tests/ because they belong
to the GPTQ subsystem and don't depend on common/.
"""
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_repro_module():
    """Dynamically import GPTQ/repro.py without requiring it on sys.path."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("gptq_repro", here / "repro.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gptq_repro"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_presets_have_required_keys():
    repro = _load_repro_module()
    assert "w4g128" in repro.PRESETS
    cfg = repro.PRESETS["w4g128"]
    for key in ("bits", "group_size", "desc_act", "sym"):
        assert key in cfg
    assert cfg["bits"] == 4
    assert cfg["group_size"] == 128


def test_parse_args_minimal(monkeypatch):
    repro = _load_repro_module()
    monkeypatch.setattr(sys, "argv", [
        "repro.py", "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "--config", "w4g128",
    ])
    args = repro.parse_args()
    assert args.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert args.config == "w4g128"
    assert args.calib_samples == 128       # default
    assert args.seq_len == 2048            # default
    assert args.eval == "ppl,zeroshot,memory"  # default


def test_parse_args_unknown_config_rejected(monkeypatch):
    repro = _load_repro_module()
    monkeypatch.setattr(sys, "argv", ["repro.py", "--model", "x", "--config", "bogus"])
    with pytest.raises(SystemExit):
        repro.parse_args()
```

- [ ] **Step 2: 跑测试确认失败**

```powershell
pytest GPTQ/test_repro_cli.py -v
```

预期：FAIL（`AttributeError: module 'gptq_repro' has no attribute 'PRESETS'`）。

- [ ] **Step 3: 修改 `GPTQ/repro.py` — 加 PRESETS 和 parse_args**

```python
"""GPTQ Phase 1 reproduction entry point.

Pipeline:
    1. Load HF model & tokenizer (via common.models)
    2. Load C4 calibration set (via common.data)
    3. Quantize with auto-gptq (algorithm-side, this file's job)
    4. Save & reload quantized model
    5. Eval PPL / zero-shot / memory (via common.eval)
    6. Write results JSON + meta JSON

Usage:
    python repro.py --model meta-llama/Llama-2-7b-hf --config w4g128

Full main() implementation lives in Task 10. This file currently provides
PRESETS and parse_args (Task 9).
"""
import argparse
import sys


PRESETS: dict[str, dict] = {
    "w4g128": {"bits": 4, "group_size": 128, "desc_act": False, "sym": True},
    "w3g128": {"bits": 3, "group_size": 128, "desc_act": False, "sym": True},
    "w2g128": {"bits": 2, "group_size": 128, "desc_act": False, "sym": True},
    "w4g-1":  {"bits": 4, "group_size": -1,  "desc_act": False, "sym": True},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="repro.py", description="GPTQ Phase 1 reproduction")
    p.add_argument("--model", required=True, help="HF model id, e.g. meta-llama/Llama-2-7b-hf")
    p.add_argument("--config", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", default="ppl,zeroshot,memory",
                   help="Comma-separated subset of {ppl,zeroshot,memory}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-quant", default=None, help="Dir to save quantized weights")
    p.add_argument("--out", default="results/")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke mode: smaller model, fewer samples (informational tag, defaults unchanged)")
    return p.parse_args(argv)


def main():
    raise NotImplementedError("Full main() implemented in Task 10 of the bootstrap plan.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑测试确认通过**

```powershell
pytest GPTQ/test_repro_cli.py -v
```

预期：3 PASSED。

- [ ] **Step 5: Commit**

```powershell
git add GPTQ/repro.py GPTQ/test_repro_cli.py
git commit -m "feat(gptq): add CLI argparse + quantization presets"
```

---

## Task 10: 实现 `GPTQ/repro.py` 端到端 main flow（无单元测试，smoke 验证）

**Files:**
- Modify: `F:\CODE\Quant\reproduce\GPTQ\repro.py`

> 整合性脚本不易做有意义的单元测试 — Task 12 的 smoke run 是真正的验证。

- [ ] **Step 1: 替换 `GPTQ/repro.py` 为完整实现**

完整文件内容：
```python
"""GPTQ Phase 1 reproduction entry point.

Pipeline:
    1. Load HF tokenizer (via common.models)
    2. Load C4 calibration set (via common.data)
    3. Quantize with auto-gptq
    4. Save & reload as quantized
    5. Eval PPL / zero-shot / memory (via common.eval)
    6. Write results JSON + meta JSON

Usage:
    python repro.py --model meta-llama/Llama-2-7b-hf --config w4g128
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

# Allow `python repro.py` from inside GPTQ/ to find the top-level common package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from common.data import load_c4_calibration, load_wikitext2_test
from common.eval.memory import measure_weight_memory, peak_gpu_memory
from common.eval.ppl import compute_ppl
from common.eval.zeroshot import evaluate_zeroshot
from common.models import load_tokenizer


PRESETS: dict[str, dict] = {
    "w4g128": {"bits": 4, "group_size": 128, "desc_act": False, "sym": True},
    "w3g128": {"bits": 3, "group_size": 128, "desc_act": False, "sym": True},
    "w2g128": {"bits": 2, "group_size": 128, "desc_act": False, "sym": True},
    "w4g-1":  {"bits": 4, "group_size": -1,  "desc_act": False, "sym": True},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="repro.py", description="GPTQ Phase 1 reproduction")
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--config", required=True, choices=list(PRESETS.keys()))
    p.add_argument("--calib-samples", type=int, default=128)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval", default="ppl,zeroshot,memory")
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-quant", default=None)
    p.add_argument("--out", default="results/")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _build_calibration_examples(tokens_list, device):
    """Convert list of 1D LongTensors into the dict shape auto-gptq's quantize() expects."""
    return [
        {
            "input_ids": t.unsqueeze(0).to(device),
            "attention_mask": torch.ones_like(t).unsqueeze(0).to(device),
        }
        for t in tokens_list
    ]


def _collect_meta(args, t_start, t_end, peak_bytes):
    import auto_gptq
    import transformers
    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "auto_gptq": auto_gptq.__version__,
        "model": args.model,
        "config": args.config,
        "preset": PRESETS[args.config],
        "calib_samples": args.calib_samples,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "device": args.device,
        "argv": sys.argv,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_peak_bytes": peak_bytes,
        "t_start": t_start,
        "t_end": t_end,
        "elapsed_sec": t_end - t_start,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_set = {x.strip() for x in args.eval.split(",") if x.strip()}
    t_start = time.time()

    # auto_gptq imported here so --help doesn't require GPU/CUDA env
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    print(f"[1/6] Loading tokenizer for {args.model}…", flush=True)
    tokenizer = load_tokenizer(args.model)

    print(f"[2/6] Loading C4 calibration ({args.calib_samples} × {args.seq_len})…", flush=True)
    calib_tokens = load_c4_calibration(
        tokenizer,
        n_samples=args.calib_samples,
        seq_len=args.seq_len,
        seed=args.seed,
    )

    preset = PRESETS[args.config]
    quant_config = BaseQuantizeConfig(
        bits=preset["bits"],
        group_size=preset["group_size"],
        desc_act=preset["desc_act"],
        sym=preset["sym"],
    )

    print(f"[3/6] Quantizing with GPTQ {args.config}…", flush=True)
    with peak_gpu_memory(args.device) as peak:
        gptq_model = AutoGPTQForCausalLM.from_pretrained(
            args.model,
            quant_config,
            torch_dtype=torch.float16,
        )
        examples = _build_calibration_examples(calib_tokens, args.device)
        gptq_model.quantize(examples)

        print(f"[4/6] Saving & reloading quantized model…", flush=True)
        save_dir = args.save_quant or str(Path(args.out) / f"quantized_{args.config}")
        gptq_model.save_quantized(save_dir)
        del gptq_model
        torch.cuda.empty_cache()
        gptq_model = AutoGPTQForCausalLM.from_quantized(save_dir, device=args.device)

    print(f"[5/6] Evaluating: {sorted(eval_set)}", flush=True)
    results: dict = {}

    if "ppl" in eval_set:
        wt2 = load_wikitext2_test(tokenizer)
        ppl = compute_ppl(
            gptq_model.model, wt2,
            seq_len=args.seq_len, stride=args.seq_len, device=args.device,
        )
        results["ppl_wikitext2"] = ppl
        print(f"  WikiText2 PPL = {ppl:.4f}", flush=True)

    if "zeroshot" in eval_set:
        zs = evaluate_zeroshot(gptq_model.model, tokenizer)
        results["zeroshot"] = zs
        for task, acc in zs.items():
            print(f"  {task} = {acc:.4f}", flush=True)

    if "memory" in eval_set:
        mem = measure_weight_memory(gptq_model.model)
        results["memory"] = mem
        print(f"  weights = {mem['weights_bytes'] / 2**20:.1f} MB, "
              f"buffers = {mem['buffers_bytes'] / 2**20:.1f} MB", flush=True)

    t_end = time.time()

    print(f"[6/6] Writing results to {out_dir}/", flush=True)
    (out_dir / f"results_{args.config}.json").write_text(json.dumps(results, indent=2))
    meta = _collect_meta(args, t_start, t_end, peak.bytes)
    (out_dir / f"meta_{args.config}.json").write_text(json.dumps(meta, indent=2, default=str))

    print(f"Done in {meta['elapsed_sec']:.1f}s. "
          f"GPU peak = {peak.bytes / 2**30:.2f} GB.", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 CLI test 确认未破坏**

```powershell
pytest GPTQ/test_repro_cli.py -v
```

预期：3 PASSED（之前的测试仍然过）。

- [ ] **Step 3: 验证 --help 工作**

```powershell
python GPTQ/repro.py --help
```

预期：argparse 输出，含 `--model`、`--config`、`--smoke` 等。**不应**因为 import auto_gptq 而失败（`from auto_gptq import ...` 在 main() 内部，--help 时不触发）。

> 如果 `--help` 因 import 失败：可能是 conda env `quant-gptq` 还没建好；先做 Task 11 的 env 安装步骤再回来。

- [ ] **Step 4: Commit**

```powershell
git add GPTQ/repro.py
git commit -m "feat(gptq): implement end-to-end quantize → eval → save pipeline"
```

---

## Task 11: 写环境管理脚本（PowerShell 本地 + bash 实验室）

**Files:**
- Create: `F:\CODE\Quant\reproduce\scripts\env_local.ps1`
- Create: `F:\CODE\Quant\reproduce\scripts\env_lab.sh`
- Create: `F:\CODE\Quant\reproduce\scripts\run_phase1_method.sh`

- [ ] **Step 1: 创建 `scripts/env_local.ps1`**

```powershell
# Set up GPTQ conda env on the local Windows machine.
# Run from repo root in PowerShell (with conda available on PATH).
#
# Usage: .\scripts\env_local.ps1 GPTQ
#
# Reuses HF_HOME (~/.cache/huggingface by default) so model weights aren't
# re-downloaded across envs.

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("GPTQ", "AWQ", "BiLLM", "KIVI")]
    [string]$Method
)

$ErrorActionPreference = "Stop"

$EnvName = "quant-$($Method.ToLower())"
$YmlPath = Join-Path $PSScriptRoot ".." $Method "env.yml"

if (-not (Test-Path $YmlPath)) {
    Write-Error "No env.yml at $YmlPath"
    exit 1
}

Write-Host "Creating conda env $EnvName from $YmlPath" -ForegroundColor Cyan
conda env create -f $YmlPath

Write-Host "Activate then install common:" -ForegroundColor Cyan
Write-Host "  conda activate $EnvName" -ForegroundColor Yellow
Write-Host "  pip install -e ." -ForegroundColor Yellow

if (-not $env:HF_HOME) {
    $defaultHfHome = Join-Path $HOME ".cache" "huggingface"
    Write-Host "Tip: set `$env:HF_HOME = `"$defaultHfHome`" to share model cache across envs."
}
```

- [ ] **Step 2: 创建 `scripts/env_lab.sh`**

```bash
#!/usr/bin/env bash
# Set up a method's conda env on the Linux lab server.
# Run from repo root: bash scripts/env_lab.sh GPTQ
#
# Lab convention: HF_HOME points to a shared cache dir
# (set via /etc/profile.d/ or per-user .bashrc, not here).

set -euo pipefail

METHOD="${1:?Usage: $0 <GPTQ|AWQ|BiLLM|KIVI>}"
ENV_NAME="quant-$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')"
YML_PATH="$(dirname "$0")/../$METHOD/env.yml"

if [[ ! -f "$YML_PATH" ]]; then
    echo "No env.yml at $YML_PATH" >&2
    exit 1
fi

echo "Creating conda env $ENV_NAME from $YML_PATH"
conda env create -f "$YML_PATH"

echo "Activate then install common:"
echo "  conda activate $ENV_NAME"
echo "  pip install -e ."

if [[ -z "${HF_HOME:-}" ]]; then
    echo "Tip: export HF_HOME=/shared/path/huggingface_cache to share model weights"
fi
```

- [ ] **Step 3: 创建 `scripts/run_phase1_method.sh`**

```bash
#!/usr/bin/env bash
# Run a method's repro.py with canonical settings on the lab server.
# Usage: bash scripts/run_phase1_method.sh GPTQ [--smoke]
#
# Assumes: conda env quant-<method> already created via env_lab.sh
# and `pip install -e .` already run inside it.

set -euo pipefail

METHOD="${1:?Usage: $0 <GPTQ|AWQ|BiLLM|KIVI> [extra-repro-args...]}"
shift || true

ENV_NAME="quant-$(echo "$METHOD" | tr '[:upper:]' '[:lower:]')"
SUBDIR="$(dirname "$0")/../$METHOD"

echo "Activating $ENV_NAME"
# `conda activate` requires shell hook
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

cd "$SUBDIR"
echo "Running repro.py in $(pwd)"
python repro.py \
    --model meta-llama/Llama-2-7b-hf \
    --config w4g128 \
    --eval ppl,zeroshot,memory \
    --out results/ \
    "$@"
```

- [ ] **Step 4: chmod +x（仅 .sh，PowerShell 不需要）**

```powershell
# Windows 上 .sh 由 Linux 端执行；本地无需 chmod。push 到 git 后在 lab 上 chmod。
# 提交时记录可执行 bit：
git update-index --chmod=+x scripts/env_lab.sh scripts/run_phase1_method.sh
```

> 如 `git update-index --chmod=+x` 在 Windows 上报错，先 `git config core.fileMode false` —— 但我们**不修改 git config**（按系统约束）。这种情况下跳过此步，commit 后到 Linux 端再 `chmod +x scripts/*.sh && git add scripts/*.sh && git commit --amend --no-edit`，由用户决定何时执行。

- [ ] **Step 5: 本地烟雾跑 `env_local.ps1`**

```powershell
.\scripts\env_local.ps1 GPTQ
```

预期：成功创建 `quant-gptq` conda env。如果已存在，跳过此步或先 `conda env remove -n quant-gptq`。

- [ ] **Step 6: 在 conda env 里装 common**

```powershell
conda activate quant-gptq
pip install -e .
```

预期：`pip install -e .` 安装 `quant-reproduce-common` 包。

- [ ] **Step 7: Commit**

```powershell
git add scripts/
git commit -m "feat(scripts): add per-method env setup (PowerShell local + bash lab)"
```

---

## Task 12: 在 12GB 本地用 TinyLlama 烟雾跑 GPTQ

**Files:**
- Modify: `F:\CODE\Quant\reproduce\GPTQ\results\` （由 repro.py 写入）

> 目标：验证 `repro.py` 端到端不报错，**不**追求论文数字。

- [ ] **Step 1: 确认 GPU 可见**

```powershell
conda activate quant-gptq
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

预期：`True <你的GPU型号>`。

- [ ] **Step 2: 烟雾跑（仅 PPL，TinyLlama-1.1B，calib=32，seq_len=1024）**

```powershell
cd GPTQ
python repro.py `
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --config w4g128 `
    --calib-samples 32 `
    --seq-len 1024 `
    --eval ppl,memory `
    --out results/smoke/ `
    --smoke
```

预期：5–15 分钟跑完。生成：
- `results/smoke/results_w4g128.json` 含 `ppl_wikitext2`（任意正浮点）和 `memory`
- `results/smoke/meta_w4g128.json` 含 torch / auto-gptq 等版本

- [ ] **Step 3: 检查输出格式合理**

```powershell
type results\smoke\results_w4g128.json
```

预期：JSON，含 `ppl_wikitext2` 字段（数值 > 0），`memory.weights_bytes` 字段（> 0）。

> **重要**：smoke 跑 PPL 数字本身不要用来判断"复现成功"。只判断是否报错 / JSON 结构正确。

- [ ] **Step 4: 把 smoke 结果加到 .gitignore（不上传）**

修改 `.gitignore`，在 "量化产物目录" 那段下追加：
```
# smoke 跑结果不进 git（仅 canonical 数字进）
*/results/smoke/
```

- [ ] **Step 5: Commit**

```powershell
git add .gitignore
git commit -m "test(gptq): smoke run with TinyLlama-1.1B passes (results not committed)"
```

---

## Task 13: 在 96GB 实验室 GPU 跑 LLaMA-2-7B canonical（人工任务）

**Files:**
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\results_w4g128.json`（由 repro.py 写入）
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\meta_w4g128.json`
- Modify: `F:\CODE\Quant\reproduce\GPTQ\README.md`（填实测数字）

> 这是人工任务，需要把 repo push 到 lab server（或拷贝文件夹），跑完再把 results/ 拉回。

- [ ] **Step 1: 在 lab server 准备 env**

```bash
# 在 lab server shell
git clone <your-repo-url> ~/quant-reproduce  # 或 rsync 整个目录过去
cd ~/quant-reproduce
bash scripts/env_lab.sh GPTQ
conda activate quant-gptq
pip install -e .
```

预期：env 创建成功；`python -c "import auto_gptq, common"` 静默通过。

- [ ] **Step 2: HF token & model 下载**

如果 LLaMA-2 需要授权，先 `huggingface-cli login` 输入 HF token。然后预下载模型避免计入跑时：

```bash
python -c "from common.models import load_hf_model; load_hf_model('meta-llama/Llama-2-7b-hf')"
```

预期：14 GB 下载到 `$HF_HOME`（lab 共享路径）。

- [ ] **Step 3: 跑 canonical**

```bash
cd GPTQ
python repro.py \
    --model meta-llama/Llama-2-7b-hf \
    --config w4g128 \
    --calib-samples 128 \
    --seq-len 2048 \
    --eval ppl,zeroshot,memory \
    --out results/
```

预期：30–90 分钟跑完。生成 `results/results_w4g128.json`（含 PPL ≈ 5.4–6.0、6 项 zero-shot、memory）和 `results/meta_w4g128.json`。

> **数字判定**：
> - `ppl_wikitext2` 落在 [5.39, 5.99]（即 5.69 ± 0.3）→ ✅ 通过 §3.4 容差
> - 落在 [4.5, 7.0] 但不含 [5.39, 5.99] → ⚠️ 同量级，但需在 GPTQ/README.md 写排查记录
> - 落在 > 10 → ❌ 怀疑 calibration / 模型 ID / preset 出错，按 GPTQ/README.md Troubleshooting 顺序排查

- [ ] **Step 4: 把 results/ 拉回本地**

```bash
# 在 lab server 上
cd ~/quant-reproduce
git add GPTQ/results/results_w4g128.json GPTQ/results/meta_w4g128.json
git commit -m "data(gptq): canonical W4-g128 results on LLaMA-2-7B"
git push
```

或 `rsync GPTQ/results/ <local>:F:/CODE/Quant/reproduce/GPTQ/results/`。

- [ ] **Step 5: 在本地拉取**

```powershell
git pull
```

预期：`GPTQ/results/results_w4g128.json` 和 `meta_w4g128.json` 出现在本地。

- [ ] **Step 6: 更新 GPTQ/README.md "实测 vs 论文" 表**

读 `GPTQ/results/results_w4g128.json` 取实测数字，把 README.md 中：
```
| w4g128 | LLaMA-2-7B | WT2 PPL | _TBD_ | ≈ 5.69 | ±0.3 |
```
替换为（举例）：
```
| w4g128 | LLaMA-2-7B | WT2 PPL | **5.71** | ≈ 5.69 | ✅ 在 ±0.3 内 |
```

zero-shot 行同样填实测。memory 行填 `weights_bytes / 2**30` 的 GB 数。

如果数字偏出 ±0.3：在 README 末尾加 "## 实测异常记录" 一节，写已排查的步骤（参考 §4.4 5 步）。

- [ ] **Step 7: Commit README 更新**

```powershell
git add GPTQ/README.md
git commit -m "docs(gptq): fill in canonical numbers vs paper anchor"
```

---

## Task 14: 写 Phase 2 GPTQ 源码研读笔记

**Files:**
- Create: `F:\CODE\Quant\reproduce\docs\reports\gptq.md`

> 按 spec §3.5 五节模板。读 `python -c "import auto_gptq; print(auto_gptq.__file__)"` 指向的 site-packages 源码。

- [ ] **Step 1: 找到 auto-gptq 源码位置**

```powershell
conda activate quant-gptq
python -c "import auto_gptq, os; print(os.path.dirname(auto_gptq.__file__))"
```

预期：例如 `~/anaconda3/envs/quant-gptq/lib/python3.10/site-packages/auto_gptq`。记下此路径，后面要用。

- [ ] **Step 2: 找到核心量化函数**

关键文件（auto-gptq 0.7+）：
- `auto_gptq/quantization/gptq.py` — 核心算法
- `auto_gptq/modeling/_base.py` — 模型加载与 quantize() orchestrator
- `auto_gptq/modeling/llama.py` — LLaMA 适配（哪些 Linear 被替换）

阅读重点：
- GPTQ 类的 `add_batch` / `fasterquant` 方法 — 主算法
- 列序选择（`actorder` / `desc_act`）
- Hessian 求逆（用 Cholesky）

- [ ] **Step 3: 创建 `docs/reports/gptq.md` 五节**

```markdown
# GPTQ 源码研读笔记（Phase 2）

> Reading auto-gptq @ `<填上面 Step 1 输出的路径>`，对照原版 IST-DASLab/gptq。

## 1. 算法回顾

GPTQ 是 layer-wise post-training quantization：每个 Linear 层独立处理，给定该层
calibration 输入 X 与权重 W，目标是找量化后 `Q(W)` 使输出 `Q(W) X` 与 `W X` 的
MSE 最小。核心公式来自 OBQ：用近似 Hessian `H = 2 X X^T` 的逆做误差补偿，
每量化一列就把误差按 Hessian 加权地传播到剩余列上。GPTQ 把 OBQ 从 O(d²) 一步
降到 O(d) 一步——按列固定顺序量化、用 Cholesky 分解避免显式求逆——这是论文
最关键的工程贡献。

## 2. 官方代码地图

入口：`AutoGPTQForCausalLM.quantize(examples)` → `_base.py` 中
`BaseGPTQForCausalLM.quantize`。

调用栈关键节点：

```
.quantize(examples)
  └ for each transformer block:
      └ for each Linear layer in block:
          └ GPTQ.add_batch(input, output)         # 累积 H = X X^T
          └ GPTQ.fasterquant(percdamp, group_size, actorder, …)
              └ Cholesky(H + λI)                  # 阻尼 + 分解
              └ for col in column_order:          # actorder 控制顺序
                  └ quantize(W[:, col])
                  └ propagate error to W[:, col+1:]
```

实际改写权重的是 `fasterquant` 内部的循环。

## 3. 关键实现选择

- **列序**：`desc_act=True` 时按 diag(H) 降序量化；`desc_act=False` 按自然顺序。
  论文证明降序累积误差更小，但代价是输出 `g_idx` 张量打乱了原始列顺序，部分
  inference kernel（如 ExLlama V1）需要重排。社区 fork 默认 `desc_act=False` 是
  推理友好取舍。
- **percdamp**：H 加阻尼项 `λ I`（默认 0.01 × mean(diag(H))）防止 Cholesky 数值
  失败。论文未细述选值；auto-gptq 用 0.01。
- **group_size**：每 `group_size` 列共享一组 (scale, zero)；-1 表示 per-channel。
  group=128 是精度/存储的常见甜点。
- **sym vs asym**：`sym=True` 时 zero=0，仅 scale；inference kernel 简单一些。

## 4. 硬件相关注释

- **累加器位宽**：算法本身用 fp32 算 H 与 Cholesky；最终量化结果存 int4 + per-group
  fp16 scale。inference 路径上累加器至少 fp16（auto-gptq 默认 ExLlama / Marlin /
  Triton kernel 都是 fp16 累加）。NPU 实现要小心：int4 × int4 累加到 int32 才是
  bit-true 的；fp16 累加器是性能取舍。
- **Kernel 形态**：auto-gptq 默认走 ExLlama V2 kernel（packed int4 → fp16
  dequant → fp16 GEMM）。这意味着**真 INT4 GEMM 没有发生**，int4 是存储格式而
  非计算格式。要做真正的 NPU-friendly INT4 计算需要 Marlin 或自写 kernel。
- **SRAM 友好度**：per-group scale 每 128 列一组，扫一行权重需要顺序读 (W_packed,
  scale, zero) 三个张量；scale tile 大小 = bs × group_size × 2 bytes，对 SRAM 不
  压力大。`desc_act=True` 时 g_idx 增加随机访问，对 cache 不友好。

## 5. 如果让我再写一遍

> 这一节是 Phase 3 unified pipeline spec 的真正素材。

- 把 `add_batch` / `fasterquant` 拆成纯函数（输入 W, H, config，输出 W_q, scales,
  zeros），与 HF 模型 hook 解耦。auto-gptq 把 orchestration 和算法绑得太死，研读
  时 callstack 来回跳很费神。
- group_size、actorder、damp 全部 dataclass 化作为 `GPTQConfig`，避免 5 个 bool/int
  参数飘来飘去。
- 与 AWQ 共享一个 `LinearReplacer` —— Phase 3 中 GPTQ 和 AWQ 都把 nn.Linear 替换为
  自己的 QuantLinear，这一步抽象出来后切换方法只是改 quantizer 实现。
- KV cache 完全不在这层 —— GPTQ 触不到。Phase 3 的 unified pipeline 需要把 weight
  quant 和 KV quant 放到不同的"挂载点"，分别 hook，不要硬塞同一个 API。
```

> 上面是模板初稿；实际写时把 `<auto-gptq path>` 填进去、引用具体文件 / 行号。

- [ ] **Step 4: 跑 markdown lint（可选）**

```powershell
# 如果装了 markdownlint-cli2 的话
markdownlint-cli2 docs/reports/gptq.md
```

预期：无错误。如未装，跳过。

- [ ] **Step 5: Commit**

```powershell
git add docs/reports/gptq.md
git commit -m "docs(reports): add Phase 2 source-code study for GPTQ"
```

---

## Task 15: 更新顶层 README + 创建 `docs/results/summary.md`（仅 GPTQ 行）

**Files:**
- Modify: `F:\CODE\Quant\reproduce\README.md`
- Create: `F:\CODE\Quant\reproduce\docs\results\summary.md`

- [ ] **Step 1: 创建 `docs/results/summary.md`（GPTQ 一行）**

```markdown
# 数字汇总（Phase 1）

> 模型：LLaMA-2-7B-hf，FP16 baseline PPL ≈ 5.47（WikiText-2 test）
> 评测协议固定：seq_len=2048, stride=2048, calib=128 × 2048 from C4 (seed=42)

| 方法 | 配置 | PPL (WT2) | piqa | arc_e | arc_c | hella | wino | obqa | weights GB | KV GB | 论文锚点 |
|------|------|-----------|------|-------|-------|-------|------|------|-----------|-------|----------|
| GPTQ | w4g128 | _<填实测>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | _<…>_ | n/a | PPL ≈ 5.69 |
| AWQ | _待跑_ | — | — | — | — | — | — | — | — | n/a | — |
| BiLLM | _待跑_ | — | — | — | — | — | — | — | — | n/a | — |
| KIVI | _待跑_ | — | — | — | — | — | — | — | — | _<…>_ | — |

## 落地数据来源

每方法的具体 `results_*.json` 与 `meta_*.json` 在各方法子目录的 `results/` 下；commit SHA 见 git log。
```

实际填入数字时从 `GPTQ/results/results_w4g128.json` 读取。

- [ ] **Step 2: 修改顶层 `README.md` 中的方法表，把 GPTQ 标记为 "已复现"**

把：
```markdown
| **GPTQ**  | Frantar et al., ICLR 2023  | 进行中 | [GPTQ/](GPTQ/) |
```
改为：
```markdown
| **GPTQ**  | Frantar et al., ICLR 2023  | ✅ 复现 + 笔记 | [GPTQ/](GPTQ/) · [研读笔记](docs/reports/gptq.md) |
```

- [ ] **Step 3: Commit**

```powershell
git add README.md docs/results/summary.md
git commit -m "docs: link GPTQ as completed in top README + initialize summary table"
```

- [ ] **Step 4: Tag 第一份里程碑**

```powershell
git tag -a phase1-gptq-done -m "Phase 1 milestone: GPTQ reproduction + Phase 2 writeup complete"
```

---

## 自检清单

实施完所有 task 后人工核对：

- [ ] `pytest tests/ -v -m "not slow"` 全过（不计 slow 标记的 lm-eval 测试）
- [ ] `pytest tests/ -v -m slow` 至少跑过一次（zero-shot 烟雾通过）
- [ ] `GPTQ/results/results_w4g128.json` 存在且 `ppl_wikitext2` 字段是合理浮点
- [ ] `GPTQ/results/meta_w4g128.json` 含完整元数据（torch/transformers/auto_gptq 版本）
- [ ] `GPTQ/README.md` 数字表已填，与论文 anchor 对照清晰
- [ ] `docs/reports/gptq.md` 五节都有内容（不只是模板）
- [ ] `docs/results/summary.md` GPTQ 行已填实数
- [ ] git log 有 ~14 个 commit（每 task 1 个左右）
- [ ] tag `phase1-gptq-done` 存在

---

## Self-Review

按 writing-plans 自检规则核对：

**1. Spec coverage**：
- spec §0.4 评测范围（PPL + zeroshot + memory）→ Tasks 5/7/6 全覆盖 ✅
- spec §1 仓库结构 → Task 1（顶层）+ Task 8（GPTQ）✅
- spec §2 common/ 接口 → Tasks 2-7 ✅
- spec §3.1 子目录模板 → Task 8 ✅
- spec §3.2 repro.py CLI → Tasks 9-10 ✅
- spec §3.3 install 策略（GPTQ pip）→ Task 11 ✅
- spec §3.4 完成判定（±0.3 PPL）→ Task 13 Step 3 判定 ✅
- spec §3.5 Phase 2 五节笔记 → Task 14 ✅
- spec §4.1 env 隔离 → Task 11 ✅
- spec §4.2 smoke vs canonical → Tasks 12 / 13 ✅
- spec §4.3 元数据落盘 → Task 10（`_collect_meta`）✅
- spec §4.4 数字打不到的 escalation → 写进 GPTQ/README.md（Task 8 Step 1）✅
- spec §4.5 Phase 1+2 完成定义 → 自检清单覆盖 ✅
- spec §5 Phase 3 占位 → 不在本 plan，将在 Phase 3 plan 中处理 ✅

**2. Placeholder scan**：
- 所有 step 都给出可直接执行的代码或命令；无 "TBD" / "implement later" / "similar to Task N"。
- 唯一例外是 GPTQ/README.md 中数字栏的 `_TBD_` 占位 — 这是**预期占位**，会在 Task 13 Step 6 填实数。

**3. Type consistency**：
- `load_wikitext2_test(tokenizer) -> torch.LongTensor` — Task 3 定义、Task 5 测试、Task 10 调用，签名一致 ✅
- `load_c4_calibration(tokenizer, n_samples, seq_len, seed)` — Task 4 定义、Task 10 调用，参数名一致 ✅
- `compute_ppl(model, tokens, seq_len, stride, device)` — Task 5 定义、Task 10 调用 ✅
- `evaluate_zeroshot(model, tokenizer, tasks, num_fewshot, batch_size, limit)` — Task 7 定义、Task 10 调用（Task 10 不传 limit，走默认 None）✅
- `measure_weight_memory(model) -> dict` — Task 6 定义返回 `{"weights_bytes", "buffers_bytes"}`，Task 10 用同样 key ✅
- `peak_gpu_memory(device)` context manager 返回 `_PeakHolder`，Task 10 通过 `peak.bytes` 访问 ✅
- `PRESETS["w4g128"]` 字典键 `bits/group_size/desc_act/sym` — Task 9 定义、Task 10 用同样键名 ✅

无类型 / 名字不一致问题。

---

## 执行交接

Plan 已存到 `docs/superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md`。两种执行方式：

**1. Subagent-Driven（推荐）** — 每 task 派一个新 subagent；任务间审查；快速迭代。
**2. Inline Execution** — 在本会话内顺序跑 task；批量执行 + checkpoint 审查。

由你选。
