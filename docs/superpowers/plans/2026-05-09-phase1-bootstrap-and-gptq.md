# Bootstrap + GPTQ Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **⚠️ Scope correction (2026-05-09)**: Tasks 1–7 (framework: `common/` + tests + scripts) and Task 11 (env scripts) were executed and committed as originally written. **Tasks 8–14 below are rewritten** after a user clarification: "Phase 1 = run upstream's reproduce scripts as-is, NOT write own `repro.py` / argparse / orchestration". The original Task 8 / 9 / 10 / 12 / 13 (which had the user write a `GPTQ/repro.py` calling `auto-gptq` API + `common/eval`) are superseded. New Tasks 8–14 below match the user's actual intent: vendor upstream → run upstream's own example scripts → capture stdout. The `common/` modules built in Tasks 2–7 stay; they're Phase 3 assets, not used in Phase 1.

**Goal:** 复现 GPTQ on LLaMA-2-7B (W4-g128) by running AutoGPTQ's own reproduce scripts to within ±0.3 PPL of the published number, then write Phase 2 source-code study notes.

**Architecture (revised)**: `GPTQ/` 子目录 vendor 上游 `AutoGPTQ` 整个 repo 进 `third_party/`，conda env `quant-gptq` 装 `auto-gptq` pip wheel + 上游 requirements，所有量化 / 评测都用上游 `examples/` 里的脚本。`docs/reports/gptq.md` 是 Phase 2 源码研读笔记。**`common/` 不在 Phase 1 import 路径上**。

**Tech Stack:** Python 3.10+，PyTorch 2.2.x，transformers，datasets，lm-eval，auto-gptq ≥ 0.7。Conda 做 env 隔离。Windows 11 (PowerShell) 本地 + Linux 实验室服务器双环境。

**Spec:** `docs/superpowers/specs/2026-05-09-quant-reproduce-design.md`
**关联文档**: `docs/howto-reproduce.md` 是给读者的操作手册（与本 plan 的"实施视角"互补）。

---

## File Structure

本计划创建/修改的文件（按目录分组）：

```
F:\CODE\Quant\reproduce\
├── pyproject.toml                       # 创建：定义 common 包 + dev tooling
├── README.md                            # 创建：项目入口 + 方法表 + 链接 GPTQ
│
├── common/                              # Phase 3 资产（Phase 1 不 import）
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
│   └── run_phase1_method.sh             # 创建：参数化跑某方法的上游脚本
│
├── GPTQ/                                # ⚠️ 修订：原 Task 8 列了 requirements.txt + repro.py，已废
│   ├── README.md                        # 创建（Task 8）：用法 + 数字 + troubleshooting
│   ├── env.yml                          # 创建（Task 8）：conda env 完整定义
│   ├── third_party/AutoGPTQ/            # 创建（Task 8）：git submodule，上游所有代码
│   └── results/                         # 跑出 stdout / quantized checkpoint 落这
│       ├── canonical_w4g128_quant_stdout.txt   # 由 Task 11 填入
│       ├── canonical_w4g128_zeroshot.json      # 由 Task 11 填入
│       ├── canonical_w4g128_zeroshot_stdout.txt
│       └── canonical_w4g128_meta.md            # 由 Task 12 填入（手记元数据）
│
└── docs/
    ├── reports/
    │   └── gptq.md                      # 创建（Task 13）：Phase 2 五节笔记
    └── results/
        └── summary.md                   # 创建（Task 14）：仅 GPTQ 一行
```

**已存在不动**：`.gitignore`，`docs/superpowers/specs/2026-05-09-quant-reproduce-design.md`。

**单文件职责**：
- `common/data.py` — 仅 dataset 加载，无算法（**Phase 3 才用到**）。
- `common/eval/*.py` — 各文件一种评测维度（**Phase 3 才用到**）。
- `common/models.py` — HF 加载唯一入口（**Phase 3 才用到**）。
- `GPTQ/third_party/AutoGPTQ/` — vendor 的上游 repo；所有量化 / 评测脚本都走它。Phase 1 不写自己的 Python 代码。
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

## Task 8: Vendor AutoGPTQ + GPTQ 子目录骨架

> ⚠️ **Phase 1 不写自己的 `repro.py` / `requirements.txt`**。Phase 1 直接用上游 `third_party/AutoGPTQ/` 的 `examples/` 跑量化 + 评测，依赖也跟着 `third_party/AutoGPTQ/requirements.txt`。`common/` 是 Phase 3 资产，Phase 1 不 import。

**Files:**
- New submodule: `F:\CODE\Quant\reproduce\GPTQ\third_party\AutoGPTQ` (via `git submodule add`)
- Modify: `F:\CODE\Quant\reproduce\.gitmodules` (auto-created)
- Create: `F:\CODE\Quant\reproduce\GPTQ\README.md`
- Create: `F:\CODE\Quant\reproduce\GPTQ\env.yml`
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\.gitkeep`

- [ ] **Step 1: Vendor AutoGPTQ via git submodule**

```powershell
cd F:\CODE\Quant\reproduce\GPTQ
git submodule add https://github.com/AutoGPTQ/AutoGPTQ third_party/AutoGPTQ
```

预期：在 `GPTQ/` 下出现 `third_party/AutoGPTQ/` 目录，包含上游全部代码；仓库根多一个 `.gitmodules` 文件（首次 submodule）。

vendor 后**立刻 `ls`** 确认上游 repo 实际目录结构（脚本路径以这里为准，不一定跟本 plan 假设的一样）：

```powershell
ls third_party\AutoGPTQ\examples\quantization\
# 期望看到 quant_with_alpaca.py（或类似命名）
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
  - pytorch=2.2.*
  - pytorch-cuda=12.1
  - pip:
    - auto-gptq>=0.7,<1.0   # pip pre-built wheel
    - sentencepiece          # LLaMA tokenizer 强制依赖
    # transformers / accelerate / datasets / lm-eval 全部走 third_party/AutoGPTQ/requirements.txt
    # Task 9 conda activate 后用 pip install -r 装，不在 env.yml 重复列。
```

- [ ] **Step 3: 创建 `GPTQ/README.md`（先填占位，跑完 canonical 后填实数）**

```markdown
# GPTQ — Phase 1 复现

> Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*, ICLR 2023
> 论文：<https://arxiv.org/abs/2210.17323>
> 上游 repo: <https://github.com/AutoGPTQ/AutoGPTQ>（社区 fork，LLaMA-2 复现事实标准）

## 上游
- vendor 路径：`third_party/AutoGPTQ`
- 用的脚本：`examples/quantization/quant_with_alpaca.py`（具体名以 vendor commit 实际为准；先 `ls third_party/AutoGPTQ/examples/` 看一眼）

## 跑法

### 一次性建 env

```powershell
# 仓库根
.\scripts\env_local.ps1 GPTQ
conda activate quant-gptq
cd GPTQ\third_party\AutoGPTQ
pip install -r requirements.txt
```

### 烟雾跑（本地 12GB / TinyLlama-1.1B / 5–10 min）

```bash
cd GPTQ/third_party/AutoGPTQ
python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --quantized_model_dir ../../results/smoke/quantized \
    --bits 4 --group_size 128 --num_samples 32
```

判定：脚本能跑完不挂；`results/smoke/quantized/` 出量化产物。**不**判定数字。

### Canonical（lab 96GB / LLaMA-2-7B / 30–90 min）

```bash
cd GPTQ/third_party/AutoGPTQ

# 量化
python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir ../../results/quantized_w4g128 \
    --bits 4 --group_size 128 --num_samples 128 \
    2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt

# 评测 zero-shot（lm-eval-harness）
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ../../results/canonical_w4g128_zeroshot.json
```

## 实测 vs 论文

| Config | Model | Metric | 实测 | 论文 anchor | 容差判定 |
|--------|-------|--------|------|-------------|---------|
| w4g128 | LLaMA-2-7B | WT2 PPL | _TBD_ | ≈ 5.69 | ±0.3 |
| w4g128 | LLaMA-2-7B | piqa | _TBD_ | _TBD_ | ±0.02 |
| w4g128 | LLaMA-2-7B | weights GB | _TBD_ | ≈ 3.7 | — |

> 数字将在跑完 canonical 后从 `results/canonical_*` 抠出。

## Troubleshooting

按 [`docs/howto-reproduce.md` §7](../docs/howto-reproduce.md) 五步排查流程。

## 算法摘要

GPTQ 用近似 Hessian 做 layer-wise 量化：每层逐列量化，每列量化后用其余列的 Hessian 信息更新剩余列以补偿误差。Cholesky 分解避免显式求逆。

详细源码研读：[`docs/reports/gptq.md`](../docs/reports/gptq.md)（Phase 2 输出）。
```

- [ ] **Step 4: 创建 `GPTQ/results/.gitkeep`**

空文件，让空目录进 git。

- [ ] **Step 5: Commit**

```powershell
cd F:\CODE\Quant\reproduce
git add .gitmodules GPTQ/
git commit -m "scaffold(gptq): vendor AutoGPTQ + env.yml + README template"
```

---

## Task 9: 建 `quant-gptq` env + 装上游依赖 + 验证

**Files:**
- 不创建新文件；只是 conda env + pip install。

- [ ] **Step 1: 建 conda env**

```powershell
cd F:\CODE\Quant\reproduce
.\scripts\env_local.ps1 GPTQ
conda activate quant-gptq
```

预期：`Successfully created environment quant-gptq`，conda 提示符前出现 `(quant-gptq)`。

- [ ] **Step 2: 装上游 requirements**

```powershell
cd F:\CODE\Quant\reproduce\GPTQ\third_party\AutoGPTQ
pip install -r requirements.txt
```

预期：装 transformers / accelerate / datasets / lm-eval 等。1–3 分钟。

- [ ] **Step 3: 三段验证**

```powershell
# 1. GPU 可见
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.2.x+cu121 True

# 2. auto-gptq 装好
python -c "import auto_gptq; print(auto_gptq.__version__)"
# 期望: 0.7.x

# 3. 上游 example 脚本能找到
ls examples\quantization\
# 期望看到 quant_with_alpaca.py 或类似（脚本名以 vendor 实际为准）

# 4. lm-eval-harness 能跑
python -m lm_eval --help | Select-Object -First 5
# 期望: usage: ... --model ... --tasks ...
```

> 若 `cuda.is_available()` 是 False：env.yml 的 `pytorch-cuda=12.1` 没生效，重装：
> ```powershell
> pip uninstall torch -y
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

- [ ] **Step 4: 不 commit**

env 状态不进 git。本 task 没有可提交内容；这是一个"基础设施验证"step。

---

## Task 10: 本地烟雾跑（上游 example + TinyLlama）

**Files:**
- Modify: `F:\CODE\Quant\reproduce\.gitignore`（追加 smoke 屏蔽规则，如已有可跳）
- 跑产物落 `GPTQ\results\smoke\`（gitignored）

- [ ] **Step 1: 检查 .gitignore 屏蔽 smoke 结果**

确认 `.gitignore` 含：
```
*/results/smoke/
```
若无，追加。

- [ ] **Step 2: 跑上游 example with TinyLlama**

```powershell
conda activate quant-gptq
cd F:\CODE\Quant\reproduce\GPTQ\third_party\AutoGPTQ

python examples\quantization\quant_with_alpaca.py `
    --pretrained_model_dir TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --quantized_model_dir ..\..\results\smoke\quantized `
    --bits 4 --group_size 128 `
    --num_samples 32 `
    2>&1 | Tee-Object ..\..\results\smoke\stdout.txt
```

> 实际参数名可能不同。先 `python examples\quantization\quant_with_alpaca.py --help` 看一眼。如果上游脚本叫别的（比如 `examples/quantize_llama.py`），相应替换。

预期：5–15 分钟跑完。

- [ ] **Step 3: 判定**

- 命令退出码 0
- `GPTQ\results\smoke\quantized\` 目录存在，含 `model.safetensors` 或类似文件
- `GPTQ\results\smoke\stdout.txt` 含进度 / 最终 "saved to ..." 的行

满足 = 本地 env 健康，可以推到 lab 跑 canonical。

> **不**关心烟雾跑的 PPL 数字 —— 这一步只看流程通不通。

- [ ] **Step 4: Commit（仅 .gitignore 改动）**

```powershell
cd F:\CODE\Quant\reproduce
git status   # smoke 产物在 gitignored 路径，不在 status 里出现
```

如果 `.gitignore` 追加了内容：

```powershell
git add .gitignore
git commit -m "test(gptq): smoke run with TinyLlama-1.1B passes (results gitignored)"
```

---

## Task 11: Lab canonical 跑（人工任务）

**Files:**
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\canonical_w4g128_quant_stdout.txt`（lab 上跑完后落 git）
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\canonical_w4g128_zeroshot.json`
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\canonical_w4g128_zeroshot_stdout.txt`

> 这是人工任务：把代码 push / rsync 到 lab；lab 上跑；results 拉回。

- [ ] **Step 1: 把代码同步到 lab**

```powershell
# 本地：把 Tasks 8 / 10 的 commit 都 push 出去
git push origin master
```

```bash
# Lab 首次 clone：
git clone --recurse-submodules <你的远端 URL> ~/quant-reproduce

# 后续 pull 含 submodule 更新：
cd ~/quant-reproduce && git pull && git submodule update --init --recursive
```

> 若忘记 `--recurse-submodules`：在 lab 上 `git submodule update --init --recursive`。

- [ ] **Step 2: Lab 上准备 env**

```bash
cd ~/quant-reproduce
bash scripts/env_lab.sh GPTQ
conda activate quant-gptq

cd GPTQ/third_party/AutoGPTQ
pip install -r requirements.txt

# 一次性预下载 LLaMA-2-7B
huggingface-cli login    # 粘贴 HF token
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')"
# 14GB 下载到 $HF_HOME
```

- [ ] **Step 3: 跑量化（30–60 min）**

```bash
cd ~/quant-reproduce/GPTQ/third_party/AutoGPTQ
nohup python examples/quantization/quant_with_alpaca.py \
    --pretrained_model_dir meta-llama/Llama-2-7b-hf \
    --quantized_model_dir ../../results/quantized_w4g128 \
    --bits 4 --group_size 128 --num_samples 128 \
    > ../../results/canonical_w4g128_quant_stdout.txt 2>&1 &

# 跟着看
tail -f ../../results/canonical_w4g128_quant_stdout.txt
```

完成标志：stdout 末尾有 `saved to ../../results/quantized_w4g128`。

- [ ] **Step 4: 跑评测（lm-eval-harness，~30 min）**

```bash
cd ~/quant-reproduce/GPTQ/third_party/AutoGPTQ
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ../../results/canonical_w4g128_zeroshot.json \
    2>&1 | tee ../../results/canonical_w4g128_zeroshot_stdout.txt
```

预期产出：`canonical_w4g128_zeroshot.json` 含 6 个任务的 `acc,none` 数值。

> 论文也跑了 PPL on WikiText-2。如果上游 `examples/` 没有现成 PPL 脚本，可以让 `lm-eval` 一并跑（`--tasks wikitext`），或参考 `docs/howto-reproduce.md` §5.4.3 自己写一段。

- [ ] **Step 5: 把 results 拉回**

```bash
# Lab 上
cd ~/quant-reproduce
git add GPTQ/results/canonical_w4g128_quant_stdout.txt \
        GPTQ/results/canonical_w4g128_zeroshot.json \
        GPTQ/results/canonical_w4g128_zeroshot_stdout.txt
git commit -m "data(gptq): canonical W4-g128 results on LLaMA-2-7B"
git push
```

```powershell
# 本地
cd F:\CODE\Quant\reproduce
git pull
ls GPTQ\results\
# 看到 canonical_*.txt / canonical_*.json
```

> 不要把 `GPTQ/results/quantized_w4g128/` 整个量化产物文件夹（~3.7 GB）push 进 git。已通过 `.gitignore` 屏蔽 `quantized_*/`。

---

## Task 12: 抽数字 + 写 README + 元数据

**Files:**
- Modify: `F:\CODE\Quant\reproduce\GPTQ\README.md`
- Create: `F:\CODE\Quant\reproduce\GPTQ\results\canonical_w4g128_meta.md`

- [ ] **Step 1: 从 stdout 抽 PPL**

```powershell
type GPTQ\results\canonical_w4g128_quant_stdout.txt | Select-String -Pattern "Perplexity|PPL"
# 或 Linux:  grep -iE "perplex|ppl" GPTQ/results/canonical_w4g128_quant_stdout.txt
```

记下 PPL 数值。如果上游量化脚本不打印 PPL，就用 lm-eval 跑 `--tasks wikitext` 出来的数字（在 zeroshot.json 里）。

- [ ] **Step 2: 从 zeroshot.json 抽 6 项 acc**

```powershell
python -c "import json; d = json.load(open('GPTQ/results/canonical_w4g128_zeroshot.json')); print({t: round(d['results'][t]['acc,none'], 4) for t in d['results']})"
```

记下 6 个数值。

- [ ] **Step 3: 量化 checkpoint 大小**

```bash
# 在 lab 上跑（量化产物只在 lab）：
du -sh ../../results/quantized_w4g128/
# 或读 stdout 里上游打印的 model size
```

- [ ] **Step 4: 填 `GPTQ/README.md` 数字表**

把 Step 1–3 的数字填进表格，例：

```markdown
| Config | Model | Metric | 实测 | 论文 anchor | 容差判定 |
|--------|-------|--------|------|-------------|---------|
| w4g128 | LLaMA-2-7B | WT2 PPL | **5.71** | ≈ 5.69 | ✅ ±0.3 内 |
| w4g128 | LLaMA-2-7B | piqa | 0.781 | ≈ 0.78 | ✅ |
| w4g128 | LLaMA-2-7B | weights GB | 3.71 | ≈ 3.7 | ✅ 真 INT4 |
```

> 数字打不到容差 → 在 README "Troubleshooting" 节按 `docs/howto-reproduce.md` §7 五步排查；**不要刷参数**。

- [ ] **Step 5: 创建 `GPTQ/results/canonical_w4g128_meta.md`**

```markdown
# canonical_w4g128 元数据

- 模型: meta-llama/Llama-2-7b-hf @ commit `<填 HF SHA>`
- GPU: NVIDIA <型号>
- 环境: torch <版本>, transformers <版本>, auto-gptq <版本>
- AutoGPTQ vendor commit: `<git -C third_party/AutoGPTQ rev-parse HEAD>` 给的 SHA
- 命令: 见 canonical_w4g128_quant_stdout.txt 第一行
- 时间: <开始> → <结束>（<耗时>）
- 论文 anchor 来源: AWQ paper Table 4（GPTQ 行）
```

> HF commit SHA: `cat ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/refs/main` 或 `huggingface-cli scan-cache`。
> AutoGPTQ vendor SHA：在 lab 上 `cd GPTQ/third_party/AutoGPTQ && git rev-parse HEAD`。

- [ ] **Step 6: Commit**

```powershell
git add GPTQ/README.md GPTQ/results/canonical_w4g128_meta.md
git commit -m "docs(gptq): fill in canonical numbers vs paper anchor + meta"
```

---

## Task 13: Phase 2 GPTQ 源码研读笔记

**Files:**
- Create: `F:\CODE\Quant\reproduce\docs\reports\gptq.md`

- [ ] **Step 1: 找 auto-gptq 源码位置**

```powershell
conda activate quant-gptq
python -c "import auto_gptq, os; print(os.path.dirname(auto_gptq.__file__))"
# 或直接读 vendor 的 GPTQ\third_party\AutoGPTQ\auto_gptq\（更适合 patch 研究）
```

关键文件（auto-gptq 0.7+）：
- `auto_gptq/quantization/gptq.py` — 核心 `GPTQ` 类的 `add_batch` / `fasterquant`
- `auto_gptq/modeling/_base.py` — `BaseGPTQForCausalLM.quantize` orchestrator
- `auto_gptq/modeling/llama.py` — LLaMA 适配（哪些 nn.Linear 被替换）

- [ ] **Step 2: 写 `docs/reports/gptq.md` 五节**

按 [设计文档 §3.5](../specs/2026-05-09-quant-reproduce-design.md#35-phase-2-研读笔记-docsreportsmethodmd-大纲) 模板：

1. **算法回顾**（1 段）— 假定读者懂 PTQ 但没读过这篇，一段话总结。
2. **官方代码地图** — 入口、主 class、调用栈树（哪行真正动权重）。
3. **关键实现选择** — 论文没明说但代码关键的细节：
   - 列序选取（`actorder` / `desc_act`）：降序量化降低累积误差，代价是 g_idx 张量
   - Cholesky 分解 vs 直接求逆：数值稳定性
   - 阻尼 `percdamp`：H 加 `λI` 防 Cholesky 失败
   - group_size：精度/存储甜点
   - sym vs asym：硬件 kernel 复杂度差异
4. **硬件相关注释** — 累加器位宽、kernel 是 fake-quant 还是 packed-int、对 NPU/SRAM 友好度。
   - auto-gptq 默认走 ExLlama V2 kernel：packed int4 → fp16 dequant → fp16 GEMM
   - 真 INT4 GEMM 没发生，int4 是存储格式而非计算格式
5. **如果让我再写一遍** — 你会怎么改 / 不要复制官方 repo 的什么决定。**这一节是 Phase 3 unified pipeline spec 的真正素材。**

> 这一步建议 1–2 个工作日。

- [ ] **Step 3: Commit**

```powershell
git add docs/reports/gptq.md
git commit -m "docs(reports): add Phase 2 source-code study for GPTQ"
```

---

## Task 14: 更新顶层 README + 创建 summary + 打 tag

**Files:**
- Modify: `F:\CODE\Quant\reproduce\README.md`
- Create: `F:\CODE\Quant\reproduce\docs\results\summary.md`

- [ ] **Step 1: 修改顶层 `README.md`**

把方法表里的 GPTQ 一行：

```markdown
| **GPTQ**  | Frantar et al., ICLR 2023  | 进行中 | [GPTQ/](GPTQ/) |
```

改为：

```markdown
| **GPTQ**  | Frantar et al., ICLR 2023  | ✅ 复现 + 笔记 | [GPTQ/](GPTQ/) · [研读笔记](docs/reports/gptq.md) |
```

- [ ] **Step 2: 创建 `docs/results/summary.md`**

```markdown
# 数字汇总（Phase 1）

> 模型：LLaMA-2-7B-hf，FP16 baseline PPL ≈ 5.47（WikiText-2 test）
> 评测协议：各方法跟着各自上游 example 的默认（不强求横向严格统一；横向比对在 Phase 3 用 common/eval 统一跑）

| 方法 | 配置 | PPL (WT2) | piqa | arc_e | arc_c | hella | wino | obqa | weights GB | KV GB | 论文锚点 |
|------|------|-----------|------|-------|-------|-------|------|------|-----------|-------|----------|
| GPTQ | w4g128 | _<填实测>_ | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ | _<>_ | n/a | PPL ≈ 5.69 |
| AWQ | _待跑_ | — | — | — | — | — | — | — | — | n/a | — |
| BiLLM | _待跑_ | — | — | — | — | — | — | — | — | n/a | — |
| KIVI | _待跑_ | — | — | — | — | — | — | — | — | _<>_ | — |

## 落地数据来源

每方法 `<METHOD>/results/canonical_*` 文件 + `canonical_*_meta.md` 元数据。git log 标签 `phase1-<method>-done` 标记每方法完成点。
```

填实 GPTQ 那行的数字（从 `GPTQ/README.md` 复制）。

- [ ] **Step 3: Commit + tag**

```powershell
git add README.md docs/results/summary.md
git commit -m "docs: link GPTQ as completed in top README + initialize summary table"
git tag -a phase1-gptq-done -m "Phase 1 milestone: GPTQ reproduction + Phase 2 writeup complete"
```

🎉 **GPTQ 完成。Plan B (AWQ) 等你 ping 我后写 —— 那时 GPTQ 实战经验和上游 examples 的脚本风格都积累了。**

---

## 自检清单

实施完所有 task 后人工核对：

- [ ] `pytest tests/ -v -m "not slow"` 全过（Tasks 2–7 framework 测试）
- [ ] `GPTQ/third_party/AutoGPTQ/` 是 git submodule 且已初始化
- [ ] `GPTQ/results/canonical_w4g128_quant_stdout.txt` 存在，含 PPL 数字
- [ ] `GPTQ/results/canonical_w4g128_zeroshot.json` 含 6 项 zero-shot acc
- [ ] `GPTQ/results/canonical_w4g128_meta.md` 含完整元数据
- [ ] `GPTQ/README.md` "实测 vs 论文" 表已填实数
- [ ] `docs/reports/gptq.md` 五节都写完（不只是模板）
- [ ] 顶层 `README.md` GPTQ 行改为 ✅
- [ ] `docs/results/summary.md` GPTQ 行填实
- [ ] tag `phase1-gptq-done` 存在

---

## Self-Review

按 writing-plans 自检规则核对（修订版）：

**1. Spec coverage**：
- spec §0.4 评测范围（PPL + zeroshot + memory）→ Task 11 跑 + Task 12 抽数字 ✅
- spec §1 仓库结构 → Task 1（顶层）+ Task 8（GPTQ vendor 后结构）✅
- spec §3.1 子目录模板 → Task 8（**已修订**：vendor 而非自写 repro.py）
- spec §3.2 repro.py CLI → **不再适用**：Phase 1 不写 repro.py。Phase 3 unified pipeline 才有 CLI。
- spec §3.3 install 策略（GPTQ pip）→ Task 9 ✅（pip wheel + 上游 requirements）
- spec §3.4 完成判定（±0.3 PPL）→ Task 12 Step 4 判定 ✅
- spec §3.5 Phase 2 五节笔记 → Task 13 ✅
- spec §4.1 env 隔离 → 已有 scripts (commit `a355baf`) ✅
- spec §4.2 smoke vs canonical → Tasks 10 / 11 ✅（用上游 example 而非自己 repro.py）
- spec §4.3 元数据落盘 → Task 12 Step 5（**已修订**：手记 markdown 而非自动 JSON）
- spec §4.4 数字打不到的 escalation → 引用 `docs/howto-reproduce.md` §7 ✅
- spec §4.5 Phase 1+2 完成定义 → 自检清单覆盖 ✅
- spec §5 Phase 3 占位 → 不在本 plan ✅

**2. Placeholder scan**：
- `GPTQ/README.md` 数字栏 `_TBD_` 占位 — Task 12 Step 4 填实数（预期占位）。
- 其余 step 都给出可直接执行的命令；上游脚本参数都标注"以 vendor 时实际为准"。

**3. 上游脚本名敏感性**（新增检查项）：
本 plan 多处假设上游脚本叫 `examples/quantization/quant_with_alpaca.py`。**实际名以 vendor 那一刻 repo 里的为准**。Task 8 Step 1 vendor 后立即 `ls third_party/AutoGPTQ/examples/` 确认；如果上游已改名（比如改成 `examples/quantize_llama.py`），相应替换 Task 10 / 11 的命令。

无类型 / 名字不一致问题。

---

## 执行交接

Plan 已存到 `docs/superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md`。两种执行方式：

**1. Subagent-Driven（推荐）** — 每 task 派一个新 subagent；任务间审查；快速迭代。
**2. Inline Execution** — 在本会话内顺序跑 task；批量执行 + checkpoint 审查。

由你选。
