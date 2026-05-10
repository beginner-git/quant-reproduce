# 复现操作手册（How to Reproduce）

> 这是项目的**第一份操作文档**。从工具链安装开始，到跑出第一份 GPTQ 数字为止，整条链路都覆盖。
> 假定读者：会用命令行，但不假定本机已有 Python / CUDA / Conda。
> 配套文档：[设计文档](superpowers/specs/2026-05-09-quant-reproduce-design.md) · [Plan A](superpowers/plans/2026-05-09-phase1-bootstrap-and-gptq.md)

---

## 目录

- [§0 总览](#0-总览)
- [§1 工具链安装（一次性）](#1-工具链安装一次性)
- [§2 仓库与账号 setup（一次性）](#2-仓库与账号-setup一次性)
- [§3 验证共享层（quant-common 跑通 pytest）](#3-验证共享层quant-common-跑通-pytest)
- [§4 通用方法工作流（每方法 6 步）](#4-通用方法工作流每方法-6-步)
- [§5 GPTQ 完整 walkthrough（第一次）](#5-gptq-完整-walkthrough第一次)
- [§6 各方法特异点（AWQ / BiLLM / KIVI）](#6-各方法特异点awq--billm--kivi)
- [§7 数字打不到目标的排查流程](#7-数字打不到目标的排查流程)
- [§8 完成判定](#8-完成判定)
- [§9 Colab 备用通道](#9-colab-备用通道)
- [§10 常见坑速查](#10-常见坑速查)

---

## §0 总览

### 0.1 三阶段 + 单方法循环

```
Phase 1 复现 ────► Phase 2 源码研读 ────► Phase 3 自写统一 pipeline
   ↑                  ↑                       ↑
  本文档覆盖         本文档覆盖              （Phase 1+2 完后另写 spec）
```

每个量化方法走同一个 5 步循环，**一次只做一个方法**：

```
GPTQ → AWQ → BiLLM → KIVI
 │
 ├─ 1. Vendor 上游 + 子目录骨架   (git submodule add <upstream>; 写 README + env.yml)
 ├─ 2. 建 conda env             (.\scripts\env_local.ps1 <METHOD> + 装上游 requirements)
 ├─ 3. 本地烟雾跑               (跑上游 example/quantize.py + TinyLlama-1.1B，5–10 min)
 ├─ 4. Lab canonical 跑         (跑同一个 example + LLaMA-2-7B，30–90 min)
 └─ 5. 抽数字 + 写 README       (从上游 stdout 抠 PPL / accuracy → 实测 vs 论文表)

之后再做 Phase 2: 源码笔记 (docs/reports/<method>.md，五节模板)
```

> **不是** 写自己的 `repro.py` / argparse / orchestration。**是** 跑上游 repo 自己的复现脚本，把数字捞出来。`common/` 是 Phase 3 资产。

### 0.2 评测协议（按各论文/上游 repo 默认）

不同方法的上游 example 用的 PPL 实现 / zero-shot tasks 可能略有差异，**Phase 1 不强求统一** —— 目标是各自跟自己论文的数字对得上。横向可比是 Phase 3 的事（用 `common/eval` 自己跑）。

每方法跑出的 stdout 文件保留进 git，三个月后想复盘 / 对照时知道当时跑了什么。

### 0.3 三档硬件分工

| 档位 | 配置 | 用途 |
|------|------|------|
| 本地 | 12GB GPU 单卡 | **烟雾跑**（跑上游 example + TinyLlama 验证 env） |
| 实验室 | 96GB 服务器 | **Canonical 跑**（LLaMA-2-7B 论文可比） |
| Colab Pro | A100 40GB | lab 不可用时备用（详见 §9） |

---

## §1 工具链安装（一次性）

> 本节假定你的本机是 Windows 11，lab 是 Linux。两侧都需要走一遍。

### 1.1 Git

**检查**：

```powershell
git --version
# 期望: git version 2.x.x
```

**没装就装**：<https://git-scm.com/download/win>，安装时勾选 "Git Bash"（提供 Bash shell，跑 .sh 用得上）。

**Linux lab 上**：通常已预装；没有就 `sudo apt install git` 或 `module load git`（依集群）。

**首次配身份**：

```powershell
git config --global user.name "你的名字"
git config --global user.email "yiminl.edu@gmail.com"
```

> 我们项目里**不会**改 git config，但首次本机配身份是必要的。

### 1.2 Anaconda 或 Miniconda

**检查**：

```powershell
conda --version
# 期望: conda 23.x 或更高
```

**没装就装**：

- Anaconda（带 GUI，~1 GB）：<https://www.anaconda.com/download>
- **Miniconda（推荐，~70 MB，命令行）**：<https://www.anaconda.com/docs/getting-started/miniconda/install>

安装时勾选 "Add Miniconda to PATH"（不推荐 Anaconda 默认设置）。装完**重开 PowerShell** 才能用 `conda`。

**初始化 PowerShell hook**（让 `conda activate` 在 PowerShell 里能用）：

```powershell
conda init powershell
# 重开 PowerShell 后，命令提示符前会出现 (base)
```

**Linux lab 上**：

```bash
# 如果没有
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# 同意 license, 选安装路径, 同意 init bash
source ~/.bashrc
conda --version
```

### 1.3 NVIDIA 驱动 + CUDA Toolkit

**前置**：你的 GPU 是 NVIDIA，且驱动版本要够新。

**检查驱动**：

```powershell
nvidia-smi
# 看右上角 "CUDA Version: 12.x"，需要 ≥ 12.1（PyTorch 2.x 的 CUDA 要求）
```

如果 `nvidia-smi` 找不到或 CUDA Version < 12.1：去 <https://www.nvidia.com/Download/index.aspx> 选你的 GPU 型号下载最新 Game Ready Driver 或 Studio Driver 装上。

**CUDA Toolkit 是否需要装？** 分情况：

- **只跑 AWQ / GPTQ**（用 pip 装 autoawq / gptqmodel 的 pre-built wheel）：**不需要本机有 CUDA Toolkit**，PyTorch 自带 runtime 就够。只要 `nvidia-smi` 能跑。
- **跑 BiLLM / KIVI**（需要本地编译自定义 CUDA extension）：**必须装 CUDA Toolkit 12.1+** 和 nvcc 编译器。

**装 CUDA Toolkit（KIVI 时再做也来得及）**：

- Windows: <https://developer.nvidia.com/cuda-12-1-1-download-archive> → Windows → x86_64 → 11 → exe (local)
- 装完检查：
  ```powershell
  nvcc --version
  # 期望: Cuda compilation tools, release 12.1, V12.1.x
  ```

**Linux lab**：通常集群已预装多个 CUDA 版本，用 `module load cuda/12.1` 加载（依集群文档）。

### 1.4 PowerShell 准备（Windows-specific）

**检查版本**：

```powershell
$PSVersionTable.PSVersion
# 期望: Major 5+ (Windows PowerShell 5.1 内置就够；7.x 也行)
```

**首次跑 .ps1 脚本要放开执行策略**：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# 弹窗选 [Y]es
# 之后跑 .\scripts\env_local.ps1 不再报 "running scripts is disabled"
```

> `RemoteSigned` 表示：本机写的 .ps1 直接跑；从网上下载的需要签名。比 `Bypass` 安全。

### 1.5 编辑器（建议 VS Code）

- 下载：<https://code.visualstudio.com/>
- 必装扩展：
  - Python（Microsoft 官方）— 支持 venv 切换、单元测试 UI、debug
  - Pylance — 类型检查
  - Ruff — 项目里已用的 linter
  - GitLens — 看 commit history 直观
- 推荐：把 VS Code 终端设为 PowerShell（默认就是），方便跟 `.\scripts\` 配合。

---

## §2 仓库与账号 setup（一次性）

### 2.1 仓库本地位置

```
F:\CODE\Quant\reproduce\
```

**情况 A：你之前已经跟我一起搭好框架（commit `b5b4663` 或之后）**：跳到 §3。

**情况 B：换台机器，从远端 clone**：

```powershell
cd F:\CODE\Quant
git clone <你的仓库 URL> reproduce
cd reproduce
git log --oneline | Select-Object -First 5
# 应能看到 framework 各 commit
```

### 2.2 HuggingFace 账号 + token

LLaMA-2-7B 需要授权才能下载。

**步骤**：

1. 去 <https://huggingface.co/join> 注册账号（免费）。
2. 去 <https://huggingface.co/meta-llama/Llama-2-7b-hf> 同意使用条款（点 "Agree and access repository"，等审批通常 < 1 小时）。
3. 去 <https://huggingface.co/settings/tokens> 创建一个 **Read 权限**的 token（不需要 Write）。复制 token（形如 `hf_xxxxxxxxxxx`）。
4. 在本机命令行登录：
   ```powershell
   pip install --upgrade huggingface_hub  # 如果尚未装
   huggingface-cli login
   # 粘贴 token，回车
   ```
5. 验证：
   ```powershell
   huggingface-cli whoami
   # 期望输出你的用户名
   ```

> 同步在 lab 服务器走一遍 step 4。

### 2.3 设置 `HF_HOME`（避免 14GB 模型下载 4 次）

`HF_HOME` 是 HuggingFace cache 根目录。每个 conda env 都用它读模型，所以**设到一个固定的大盘路径**，跨 env 共享。

**Windows**：

```powershell
# 永久（用户级），重开 PowerShell 才生效
[Environment]::SetEnvironmentVariable("HF_HOME", "D:\hf_cache", "User")

# 或临时（仅当前会话）
$env:HF_HOME = "D:\hf_cache"
```

**Linux lab**：

```bash
# 加到 ~/.bashrc 末尾
export HF_HOME=/shared/path/huggingface_cache
# 重开 shell 或 source ~/.bashrc 生效
echo $HF_HOME
```

> **检查空间**：LLaMA-2-7B 约 14 GB，再加 WikiText-2 + C4 + 量化产物，预留 30 GB 起步。
> **不要把 HF_HOME 设到 SSD 系统盘**（OneDrive 同步路径也避免 — 否则 14GB 会被同步上传）。

---

## §3 验证共享层（`quant-common` 跑通 pytest）

> 跑 4 个方法之前，先做一遍这个"对账"，确认 `common/` 在你环境里能用。

### 3.1 创建专用 dev env

不要污染 anaconda `base`。建一个 `quant-common`：

```powershell
conda create -n quant-common python=3.10 -y
conda activate quant-common
```

**为什么 Python 3.10**：项目 `pyproject.toml` 写了 `requires-python = ">=3.10"`。3.10 / 3.11 / 3.12 都行；3.13 也能跑但有些上游库（gptqmodel、lm-eval、flash-attn 等）的 wheel 还没全跟上，3.10 最稳。

### 3.2 装 framework 依赖

```powershell
cd F:\CODE\Quant\reproduce
pip install -e .[dev]
```

预期：

- 装 torch、transformers、datasets、accelerate、lm-eval、pytest 等
- 总下载 1–3 GB，2–5 分钟
- 最后一行 `Successfully installed ... quant-reproduce-common-0.1.0 ...`

> 此时 torch 应是 GPU 版（pip 自动选）。验证：
> ```powershell
> python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
> # 期望: 2.x.x+cu121 True
> ```
> 如果是 CPU 版（`+cpu`）或 `False`：先跑
> ```powershell
> pip uninstall torch -y
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

### 3.3 跑单元测试

```powershell
pytest tests/ -v -m "not slow"
```

预期：

```
tests/test_data.py::test_load_wikitext2_returns_1d_long_tensor PASSED
tests/test_data.py::test_load_wikitext2_deterministic PASSED
tests/test_data.py::test_load_c4_calibration_returns_list_of_long_tensors PASSED
tests/test_data.py::test_load_c4_calibration_seeded_deterministic PASSED
tests/test_data.py::test_load_c4_calibration_seed_changes_output PASSED
tests/test_eval_memory.py::test_measure_weight_memory_returns_dict_with_bytes PASSED
tests/test_eval_memory.py::test_measure_weight_memory_dtype_scaling PASSED
tests/test_eval_memory.py::test_peak_gpu_memory_records_increase PASSED   ← CUDA 版才会过；CPU 版 SKIPPED
tests/test_eval_ppl.py::test_compute_ppl_finite_and_positive PASSED        ← CUDA 版才会过
tests/test_eval_ppl.py::test_compute_ppl_deterministic PASSED              ← 同上
tests/test_eval_ppl.py::test_compute_ppl_short_input_raises PASSED         ← 同上
tests/test_models.py::test_load_tokenizer_returns_tokenizer PASSED
tests/test_models.py::test_load_hf_model_returns_model PASSED
tests/test_models.py::test_load_hf_model_respects_dtype PASSED

==================== 14 passed (or 11 passed + 3 skipped on CPU) ====================
```

第一次跑会下载几个小 dataset（WT2 ~5MB、C4 几个文档 ~1MB、tiny-gpt2 ~5MB），15–30 秒。

**通过 = 共享层在你环境里 OK**，可以开始 §5（GPTQ）了。

> Q：lm-eval 的 smoke 测试呢？
> A：默认不跑（标记为 `slow`）。等真做 GPTQ canonical 时再跑：
> ```powershell
> pytest tests/test_eval_zeroshot.py -v -m slow
> ```

### 3.4 各方法 conda env 要求总览

每个方法**独立 conda env**（命名 `quant-<method>`），不与其他方法或 base 混。原因：

- transformers 版本互斥（BiLLM/KIVI 跟着论文当时 pin 在 4.36.x；AWQ 要 ≥4.42）
- torch 主版本不同（BiLLM/KIVI 用 2.1.x；AWQ 要 ≥2.2；GPTQ 较灵活）
- 自定义 CUDA kernel（KIVI）编译时与 torch 自带 CUDA runtime 绑死，错配直接编译失败

下表是上游官方 repo 实际要求的归纳（截至 2026-01；具体以你 vendor 后看到的 `requirements.txt` 为准）：

| 顺序 | 方法 | conda env name | Python | PyTorch | CUDA | 主依赖（pip） | 装法 | 上游来源 |
|------|------|---------------|--------|---------|------|--------------|------|---------|
| 1 | **AWQ**  | `quant-awq`   | 3.10 | 2.3.x | 12.1 | `autoawq>=0.2`、`autoawq-kernels`、`transformers>=4.42` | pip pre-built wheel | <https://github.com/casper-hansen/AutoAWQ> |
| 2 | **BiLLM**| `quant-billm` | 3.10 | 2.1.x | 12.1 | (vendor) `transformers~=4.36`、`accelerate` | git submodule + `pip install -e third_party/BiLLM` | <https://github.com/Aaronhuang-778/BiLLM> |
| 3 | **KIVI** | `quant-kivi`  | 3.10 | 2.1.x | 12.1 | (vendor) `transformers~=4.36`、`flash-attn>=2.4`、`triton` | git submodule + 编译 `setup_cuda.py` | <https://github.com/jy-yuan/KIVI> |
| 4 | **GPTQ** | `quant-gptq`  | 3.10 | 2.2.x | 12.1 | `gptqmodel>=7.0`、`lm-eval` | pip pre-built wheel；上游是 library 无 examples，自写 ~10 行 quickstart | <https://github.com/ModelCloud/GPTQModel>（[AutoGPTQ archive 2025-04](https://github.com/AutoGPTQ/AutoGPTQ) 的继任） |

> ⚠️ 这些版本号是**起步模板**，不是最终钦点。每方法 vendor 或装完之后，对照上游 README / `requirements.txt` 检查一遍：上游可能又改了 pin。如果 vendor 后 `pip install -r third_party/<repo>/requirements.txt` 报版本冲突，**优先按 vendor repo 的 pin** 调你的 `<METHOD>/env.yml`。

### 3.5 每个方法的 env 都要的"公共底座"

不管哪个方法，下面这套必须在 env 里：

```yaml
# 每个 <METHOD>/env.yml 都长这样
name: quant-<method>
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch=2.x.*           # 各方法不同，见 §3.4 表
  - pytorch-cuda=12.1       # 与上面 torch CUDA 版必须一致
  - pip:
    - -r requirements.txt   # 该方法的 pip 依赖
```

```text
# 每个 <METHOD>/requirements.txt 都至少含这些（公共部分）
transformers>=4.40           # 各方法可能 pin 不同的版本
accelerate>=0.27
datasets>=2.18
sentencepiece                # LLaMA / Mistral tokenizer 必须
lm-eval>=0.4.2               # 用于 zero-shot 评测
# ↓ 加各方法专属的 pip 包 ↓
```

**Phase 1 不需要在每个方法 env 里装 `common/`** —— Phase 1 跑的是上游 repo 自己的脚本，不 import `common/`。`pip install -e .` 这一步留给 Phase 3（那时你的 unified pipeline 才会 `from common.* import`）。

> 真正需要在每个方法 env 里装的是该方法 vendor 的上游 repo 自己的 `requirements.txt`：
> ```bash
> conda activate quant-<method>
> cd <METHOD>/third_party/<UPSTREAM_REPO>
> pip install -r requirements.txt   # 跟着上游 README
> ```
>
> 共享的不是 `env`（每个 env 都独立），共享的是 **`HF_HOME` 的模型 cache**（§2.3）—— 模型权重不重复下载。

---

## §4 通用方法工作流（每方法 5 步）

> **核心理念**：Phase 1 不写自己的 Python orchestration（`repro.py`、argparse、main flow 都不写）。直接跑上游 repo 自己提供的复现脚本。`common/`（已搭好的 5 个模块）是 Phase 3 资产，Phase 1 不用。

### Step 1 — Vendor 上游 repo + 写子目录骨架

```
GPTQ/                            （AWQ/ BiLLM/ KIVI/ 同结构）
├── README.md                    # 跑法（粘贴上游命令） + 数字表 + 与论文对比 + troubleshooting
├── env.yml                      # conda env 定义（pytorch + 该方法依赖）
├── third_party/<UPSTREAM_REPO>/ # git submodule，整个上游 repo 在这里
└── results/                     # 上游脚本的输出（stdout 重定向、JSON、checkpoint 路径等）
    ├── smoke/                   # 本地小模型烟雾跑（gitignored）
    ├── canonical_w4g128_stdout.txt
    ├── canonical_w4g128_meta.md  # 你手记的：HF SHA、GPU 型号、torch 版本、命令
    └── ...
```

**没有 `repro.py`，没有 `requirements.txt` 单独维护**（pip 依赖直接走 `third_party/<repo>/requirements.txt`）。

### Step 2 — 建 conda env + 装上游

```powershell
# Windows 本地
.\scripts\env_local.ps1 <METHOD>
conda activate quant-<method>

# 装上游：先按 third_party 里 requirements.txt
cd F:\CODE\Quant\reproduce\<METHOD>\third_party\<UPSTREAM_REPO>
pip install -r requirements.txt
pip install -e .                         # 让上游可以 import（如果是 vendor 型）
# 或 pip pre-built wheel 型（AWQ / GPTQ）：env.yml 已 pip 段装过了
# pip install autoawq>=0.2 autoawq-kernels      # AWQ
# pip install gptqmodel>=7.0                    # GPTQ（用 GPTQModel，非 auto-gptq）
```

> Lab 端用 `bash scripts/env_lab.sh <METHOD>` 替代第一行。

**验证上游能跑**：每个上游 repo 都有 README，里面通常有一句"快速开始"命令。先拿那个跑通，证明 env 没问题。

### Step 3 — 本地烟雾跑（用上游脚本 + 小模型）

读上游 README 或 `examples/`，找到复现脚本。比如：

| 方法 | 上游复现脚本（示例位置） |
|------|------------------------|
| AWQ  | `third_party/AutoAWQ/examples/quantize.py`（+ `examples/eval.py` 如有） |
| BiLLM | `third_party/BiLLM/run.py` |
| KIVI | `third_party/KIVI/eval/ppl_eval.py` 之类（vendor 时 `ls eval/` 看实际） |
| GPTQ | **没有 examples/**（GPTQModel 是 library）。Phase 1 自写 `GPTQ/quant_eval.py`，原样 copy GPTQModel README quickstart |

> 实际脚本名以你 vendor 那一刻 repo 里的为准。先 `ls third_party/<repo>/` 看一眼。

**烟雾跑** = 用 TinyLlama-1.1B 跑同一个脚本，验证你的 env 不挂：

```powershell
cd F:\CODE\Quant\reproduce\AWQ\third_party\AutoAWQ
python examples/quantize.py `
    --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --quant_path ../../results/smoke/quantized `
    --w_bit 4 --q_group_size 128            # 跟着上游脚本的实际参数名
```

**判定**：脚本能跑完不挂，输出一个 quantized model 文件夹。**不**判定数字。

### Step 4 — Lab canonical 跑（用上游脚本 + LLaMA-2-7B）

把代码 sync 到 lab（`git push` / `rsync`），换成 canonical 参数：

```bash
cd ~/quant-reproduce/AWQ/third_party/AutoAWQ
python examples/quantize.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --quant_path ../../results/quantized_w4g128 \
    --w_bit 4 --q_group_size 128 \
    2>&1 | tee ../../results/canonical_w4g128_stdout.txt
```

跑完后用 lm-eval-harness 一站式出 PPL + zero-shot：

```bash
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks wikitext,piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --output_path ../../results/canonical_w4g128_eval.json
```

数字会在 stdout 和 `_eval.json` 里。把全部 stdout 重定向到 `<METHOD>/results/canonical_*.txt` 留底。

### Step 5 — 写 README 数字 + 元数据

- 从 `results/canonical_*.txt` 抽出 PPL / zero-shot / memory 数字。
- 在 `<METHOD>/README.md` 填数字表（实测 vs 论文）。
- 在 `<METHOD>/results/canonical_w4g128_meta.md` 手记一份元数据（不是 JSON，就一段 markdown）：
  ```markdown
  - 模型: meta-llama/Llama-2-7b-hf @ commit abc123
  - GPU: NVIDIA A100-SXM4-40GB
  - torch: 2.3.1+cu121
  - transformers: 4.42.0
  - autoawq: 0.2.x（或方法对应的库版本）
  - 命令: python examples/quantize.py --model_path ... （示例；以方法为准）
  - 时间: 2026-05-15 14:32 → 15:47 (1h15m)
  ```
- Commit。

> Phase 2 源码笔记是另一步（按 [设计文档 §3.5](superpowers/specs/2026-05-09-quant-reproduce-design.md#35-phase-2-研读笔记-docsreportsmethodmd-大纲) 五节模板）写 `docs/reports/<method>.md`。Phase 1 不含。

---

## §5 AWQ 完整 walkthrough（第一次）

> **修订 2026-05-10**：本节原是 GPTQ walkthrough（GPTQ 当时是 Phase 1 第一个）。AutoGPTQ 死后顺序改为 **AWQ → BiLLM → KIVI → GPTQ**，所以现在 §5 用 AWQ 当详细模板。BiLLM / KIVI / GPTQ 沿用同一 5 步骨架，差异在 §6。

> **核心理念**：本节全部是"跑上游 repo 自己的脚本"。**不**写 `repro.py`、**不**写 argparse、**不**用 `common/eval`。`common/` 是 Phase 3 才用。

### §5.1 Vendor AutoAWQ + 写子目录骨架

```powershell
cd F:\CODE\Quant\reproduce\AWQ
git submodule add https://github.com/casper-hansen/AutoAWQ third_party/AutoAWQ
```

子目录极简（产物全靠手写或上游 stdout）：

| 文件/目录 | 作用 |
|----------|------|
| `AWQ/README.md` | 跑法（粘贴上游命令）+ 数字表 + 对论文比 + troubleshooting |
| `AWQ/env.yml` | conda env 定义（pytorch + autoawq pip pre-built wheel） |
| `AWQ/third_party/AutoAWQ/` | git submodule，**所有算法 / examples / eval 脚本都在这** |
| `AWQ/results/` | 跑出来的产物（stdout 重定向、quantized checkpoint、手写 meta） |

#### `AWQ/env.yml` 模板

```yaml
name: quant-awq
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pytorch=2.3.*           # autoawq 0.2 要求 torch ≥ 2.2
  - pytorch-cuda=12.1
  - pip:
    - autoawq>=0.2          # 主算法（pip pre-built wheel）
    - autoawq-kernels       # 推理 kernel（让 W4-g128 真用 INT4 GEMM 算）
    - sentencepiece         # LLaMA tokenizer 强制依赖
    - lm-eval>=0.4.2        # 跑量化后 zero-shot
    # 其它依赖跟着 third_party/AutoAWQ/requirements.txt 走
```

#### `AWQ/README.md` 起步模板（先填占位，跑完 §5.5 再填实数）

```markdown
# AWQ — LLaMA-2-7B 复现（Phase 1 第一个方法）

## 上游
- Repo: https://github.com/casper-hansen/AutoAWQ
- Vendor 路径: `third_party/AutoAWQ`
- 用的脚本: `examples/quantize.py`（看 vendor 的实际目录）

## 跑法

### 烟雾跑（本地 12GB / TinyLlama-1.1B）
```bash
cd third_party/AutoAWQ
python examples/quantize.py \
    --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --quant_path ../../results/smoke/quantized \
    --w_bit 4 --q_group_size 128
```

### Canonical（lab 96GB / LLaMA-2-7B）
```bash
cd third_party/AutoAWQ
python examples/quantize.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --quant_path ../../results/quantized_w4g128 \
    --w_bit 4 --q_group_size 128 \
    2>&1 | tee ../../results/canonical_w4g128_stdout.txt

# 评测（lm-eval-harness 一并出 PPL + zero-shot）
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks wikitext,piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --output_path ../../results/canonical_w4g128_eval.json
```

## 实测 vs 论文
| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| w4g128 | LLaMA-2-7B | WT2 PPL | _TBD_ | ≈ 5.60 | _TBD_ |
| w4g128 | LLaMA-2-7B | piqa    | _TBD_ | _TBD_  | _TBD_ |
| w4g128 | LLaMA-2-7B | weights | _TBD_ | ≈ 3.7 GB | _TBD_ |

## Troubleshooting
（数字打不到 ±0.3 时按 §7 排查流程的记录）
```

**完成后**：

```powershell
cd F:\CODE\Quant\reproduce
git add .gitmodules AWQ/
git commit -m "scaffold(awq): vendor AutoAWQ + env.yml + README template"
```

### §5.2 建 `quant-awq` conda env + 装上游依赖

```powershell
cd F:\CODE\Quant\reproduce
.\scripts\env_local.ps1 AWQ
conda activate quant-awq

# autoawq 已通过 env.yml 装好；上游 requirements 补一遍以防有遗漏
cd AWQ\third_party\AutoAWQ
pip install -r requirements.txt
```

**验证 env**：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.3.x+cu121 True

python -c "import awq; print(awq.__version__)"
# 期望: 0.2.x

# 看上游 examples/ 实际有什么脚本
ls examples\
# 期望看到 quantize.py 或类似。脚本名以 vendor 实际为准。
```

### §5.3 本地烟雾跑（用上游 example，TinyLlama）

```powershell
conda activate quant-awq
cd F:\CODE\Quant\reproduce\AWQ\third_party\AutoAWQ
python examples\quantize.py `
    --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
    --quant_path ..\..\results\smoke\quantized `
    --w_bit 4 --q_group_size 128 `
    2>&1 | Tee-Object ..\..\results\smoke\stdout.txt
```

> 实际参数名以 `python examples\quantize.py --help` 输出为准 —— autoawq 版本之间可能微调。

**判定**：脚本能跑完不挂，`AWQ/results/smoke/quantized/` 出量化后的 checkpoint。**不**关心 PPL。

> Smoke 结果默认不进 git（`.gitignore` 屏蔽 `*/results/smoke/`）。

### §5.4 Lab canonical 跑（上游 example + LLaMA-2-7B）

#### §5.4.1 把代码同步到 lab

**Option A — git push/pull**（推荐）：

```powershell
# 本地
cd F:\CODE\Quant\reproduce
git push origin master
```

```bash
# Lab 上
cd ~
git clone --recurse-submodules <你的远端 URL> quant-reproduce
# 注意 --recurse-submodules，把 third_party/AutoAWQ 一并拉下来
cd quant-reproduce
```

> 忘记 `--recurse-submodules`：`git submodule update --init --recursive`。

**Option B — rsync**：

```bash
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='*/results/smoke' \
  user@local:F:/CODE/Quant/reproduce/  ~/quant-reproduce/
```

#### §5.4.2 Lab 上准备 env

```bash
cd ~/quant-reproduce
bash scripts/env_lab.sh AWQ
conda activate quant-awq

cd AWQ/third_party/AutoAWQ
pip install -r requirements.txt

# 预下载 LLaMA-2-7B
huggingface-cli login    # 粘贴 HF token
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')"
# 14GB 下载到 $HF_HOME
```

#### §5.4.3 跑 canonical（量化 + 评测）

```bash
cd ~/quant-reproduce/AWQ/third_party/AutoAWQ

# 1. 量化（~30 min）
python examples/quantize.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --quant_path ../../results/quantized_w4g128 \
    --w_bit 4 --q_group_size 128 \
    2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt

# 2. 评测（lm-eval 一站式，PPL + 6 项 zero-shot，~30 min）
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks wikitext,piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ../../results/canonical_w4g128_eval.json \
    2>&1 | tee ../../results/canonical_w4g128_eval_stdout.txt
```

> 上游 examples/ 里如果有自带的 PPL 脚本（比如 `examples/eval.py`），可以替代 lm-eval 那一步。**以你 vendor 时实际有什么脚本为准。**

**完成标志**：`AWQ/results/canonical_w4g128_*` 系列文件都生成，含 PPL + 6 项 zero-shot 数字。

#### §5.4.4 把 `results/` 拉回本地

```bash
# Lab 上
cd ~/quant-reproduce
git add AWQ/results/canonical_w4g128_*.txt AWQ/results/canonical_w4g128_eval.json
git commit -m "data(awq): canonical W4-g128 results on LLaMA-2-7B"
git push
```

```powershell
# 本地
cd F:\CODE\Quant\reproduce
git pull
ls AWQ\results\
```

### §5.5 抽数字 + 写 README + 手记元数据

从 `eval.json` 一站抽 PPL + 6 项 acc：

```powershell
python -c "
import json
d = json.load(open('AWQ/results/canonical_w4g128_eval.json'))['results']
print('wikitext PPL:', round(d['wikitext']['word_perplexity,none'], 4))
for t in ['piqa','arc_easy','arc_challenge','hellaswag','winogrande','openbookqa']:
    print(t, ':', round(d[t]['acc,none'], 4))
"
```

填到 `AWQ/README.md`：

```markdown
| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| w4g128 | LLaMA-2-7B | WT2 PPL | **5.62** | ≈ 5.60 | ✅ ±0.3 内 |
| w4g128 | LLaMA-2-7B | piqa | 0.792 | ≈ 0.79 | ✅ |
| w4g128 | LLaMA-2-7B | weights GB | 3.71 | ≈ 3.7 | ✅ 真 INT4 packed |
```

**手记元数据**到 `AWQ/results/canonical_w4g128_meta.md`（markdown 一段就够，不写 JSON）：

```markdown
# canonical_w4g128 元数据（AWQ）

- 模型: meta-llama/Llama-2-7b-hf @ commit `<HF SHA>`
- GPU: NVIDIA <型号>
- 环境: torch <版本>, transformers <版本>, autoawq <版本>
- AutoAWQ vendor commit: `<git -C third_party/AutoAWQ rev-parse HEAD>` 给的 SHA
- 完整命令: 见 canonical_w4g128_quant_stdout.txt 第一行
- 时间: <开始> → <结束>（<耗时>）
- 论文 anchor 来源: AWQ paper Table 4
```

> HF commit SHA：`huggingface-cli scan-cache` 或 `cat ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/refs/main`。

> 数字打不到 ±0.3 → §7 排查流程。**不要为了好看刷参数。**

```powershell
git add AWQ/README.md AWQ/results/canonical_w4g128_meta.md
git commit -m "docs(awq): fill in canonical numbers vs paper anchor + meta"
```

### §5.6 Phase 2 源码笔记

```powershell
conda activate quant-awq
python -c "import awq, os; print(os.path.dirname(awq.__file__))"
# 或直接读 vendor: AWQ\third_party\AutoAWQ\awq\
```

关键文件（autoawq 0.2+）：
- `awq/quantize/quantizer.py` — 核心 AWQ 量化（搜 `scale` / `clip` 启发式）
- `awq/models/base.py` — `AutoAWQForCausalLM.quantize` orchestrator
- `awq/models/llama.py` — LLaMA 适配
- `awq/modules/linear/` — INT4 GEMM kernel 后端

按 [设计文档 §3.6](superpowers/specs/2026-05-09-quant-reproduce-design.md#36-phase-2-研读笔记-docsreportsmethodmd-大纲) 五节模板写：

1. **算法回顾**（1 段）— activation-aware salient-channel scaling 的核心
2. **官方代码地图** — 从 `quantize()` 到真正动 weights 的调用栈
3. **关键实现选择** — scale 网格搜索粒度 / clip 启发式 / per-channel vs per-tensor 的混搭
4. **硬件相关注释** — autoawq-kernels 的 GEMM/GEMV 两种 kernel 选择，对 NPU 友好度
5. **如果让我再写一遍** — Phase 3 unified pipeline 的素材

```powershell
git add docs/reports/awq.md
git commit -m "docs(reports): add Phase 2 source-code study for AWQ"
```

### §5.7 收尾 + Plan B 时机

```powershell
# 顶层 README.md 把 AWQ 行改成 "✅ 复现 + 笔记"
# 创建 docs/results/summary.md（AWQ 一行）

git add README.md docs/results/summary.md
git commit -m "docs: link AWQ as completed + initialize summary table"
git tag -a phase1-awq-done -m "Phase 1 milestone: AWQ reproduction + Phase 2 writeup complete"
```

🎉 **第一份方法完成。Plan B (BiLLM) 等你 ping 我后写 —— 那时 AWQ 实战经验已成。**

---

## §6 各方法特异点（BiLLM / KIVI / GPTQ）

AWQ 之后三个方法都按 §5 套路：vendor → 建 env → 跑上游脚本 → 抓数字 → 写 README。**唯一差异**是上游脚本路径、参数名、论文 anchor 数字、装法（KIVI 要编译，GPTQ 没 examples）。

### §6.1 BiLLM

- **Vendor**（**官方 repo 不在 PyPI，必须 vendor**）：
  ```bash
  cd BiLLM
  git submodule add https://github.com/Aaronhuang-778/BiLLM third_party/BiLLM
  ```
- **`BiLLM/env.yml`**：
  ```yaml
  name: quant-billm
  channels:
    - pytorch
    - nvidia
    - conda-forge
  dependencies:
    - python=3.10
    - pip
    - pytorch=2.1.*            # BiLLM 论文 repo 测过的版本
    - pytorch-cuda=12.1
    - pip: []
      # 实际依赖以 third_party/BiLLM/requirements.txt 为准；vendor 后用 pip install -r 装
  ```
- **vendor 后装依赖**：
  ```powershell
  conda activate quant-billm
  cd F:\CODE\Quant\reproduce\BiLLM\third_party\BiLLM
  pip install -r requirements.txt
  ```
- **上游复现脚本**：BiLLM 通常有总入口 `run.py`。**跑法**：
  ```bash
  cd BiLLM/third_party/BiLLM
  python run.py \
      --model meta-llama/Llama-2-7b-hf \
      --eval_ppl \
      --braq \
      --salient_metric hessian \
      2>&1 | tee ../../results/canonical_billm_stdout.txt
  ```
  > 上游参数风格继承自 GPTQ 原始 repo（这就是为什么 BiLLM 是 GPTQ 之前的好"预演"）；`--braq` `--salient_metric` 是 BiLLM 特有的。**以上游 README 给的命令为准。**
- **论文 anchor**: LLaMA-2-7B / 实测 PPL **20–60 都算"同量级"**（baseline 5.47，二值化容忍度大）—— 见 [设计文档 §3.5](superpowers/specs/2026-05-09-quant-reproduce-design.md#35-复现完成判定与论文同量级)。
- **跑时间**：1–4 小时（最长）；**不要在 12GB 本地跑 canonical**，OOM。
- **常见冲突**：BiLLM 的 `requirements.txt` 经常 pin 较老的 transformers，跟新版 lm-eval 可能不兼容 —— 让 BiLLM 跑出 PPL 后，单独用别的 env（比如 quant-awq）跑 lm-eval-harness 的 zero-shot 评测，间接绕开。

### §6.2 KIVI（流程最不同）

- **不需要 calibration**。KIVI 是 inference-time KV cache 量化（per-channel keys + per-token values, 2bit），跟权重量化是两个层面。
- **vendor + 编译 CUDA extension**（必须本机有 CUDA Toolkit + nvcc，§1.3）：
  ```bash
  cd KIVI
  git submodule add https://github.com/jy-yuan/KIVI third_party/KIVI
  ```
- **`KIVI/env.yml`**：
  ```yaml
  name: quant-kivi
  channels:
    - pytorch
    - nvidia
    - conda-forge
  dependencies:
    - python=3.10
    - pip
    - pytorch=2.1.*            # KIVI 官方 README 指定 2.1.x
    - pytorch-cuda=12.1        # 必须与下面 nvcc 12.1 一致
    - cuda-toolkit=12.1        # 提供 nvcc 用于编译自定义 kernel（lab 上可走 module load 替代）
    - pip:
      - -r requirements.txt
  ```
- **vendor 后装依赖 + 编译 CUDA kernel**：
  ```bash
  conda activate quant-kivi
  cd ~/quant-reproduce/KIVI/third_party/KIVI
  pip install -r requirements.txt                  # 装 transformers / datasets / flash-attn 等
  cd quant && python setup_cuda.py install         # ← 编译，~5–10 分钟
  ```
  - **常见编译失败**：nvcc 版本与 PyTorch 自带 CUDA runtime 不匹配。检查：
    ```bash
    nvcc --version          # 编译器
    python -c "import torch; print(torch.version.cuda)"   # PyTorch 期望
    ```
    主版本号要一致（都 12.1 或都 11.8）。conda env 里 `pytorch-cuda=12.1` + `cuda-toolkit=12.1` 同 channel 装就一致。
- **上游复现脚本**：KIVI 的脚本通常在 `third_party/KIVI/` 根目录或 `eval/`：
  - `pred.py` / `example_chat.py` — 推理 demo
  - `eval/ppl_eval.py` — PPL 评测（具体名以 vendor 时为准）
- **跑法**：
  ```bash
  cd KIVI/third_party/KIVI
  python eval/ppl_eval.py \
      --model_name meta-llama/Llama-2-7b-hf \
      --k_bits 2 --v_bits 2 --k_group_size 32 --v_group_size 32 \
      --use_kivi True \
      2>&1 | tee ../../results/canonical_kivi2_stdout.txt
  ```
  > 上游脚本名以 vendor 时为准；KIVI README 通常给"Reproducing the paper"段落。
- **KIVI 跑法的本质差异**：
  - **不需要** calibration（inference-time 量化）
  - **不需要** 量化后保存 checkpoint（KIVI monkey-patch 模型的 attention 层，没有"量化产物"概念）
  - 脚本就是"加载模型 → 打补丁 → 跑 PPL"一步到位
- **关键评测差异**：还要测 KV cache 显存（不只权重）。上游脚本通常自带；如果没有，自己 `nvidia-smi` 在跑 long-context PPL 时记一下峰值。Phase 3 的 `common/eval/memory.py` 会补 `measure_kv_cache_bytes`。
- **论文 anchor**: LLaMA-2-7B + KV-2bit / WT2 PPL 几乎不掉（容差 ±0.3）。

### §6.3 GPTQ（最后做，用 GPTQModel 替代死掉的 AutoGPTQ）

> ⚠️ **2026-05 上游变化**：[AutoGPTQ 已 archive](https://github.com/AutoGPTQ/AutoGPTQ)（2025-04，最后版本 0.7.1 / 2024-03）；transformers 也已 deprecate auto-gptq 后端。继任者是 [GPTQModel](https://github.com/ModelCloud/GPTQModel)，AutoGPTQ 的活跃 fork/refactor，已 merge 进 transformers/optimum/peft。本节用 GPTQModel。

- **Vendor**：
  ```bash
  cd GPTQ
  git submodule add https://github.com/ModelCloud/GPTQModel third_party/GPTQModel
  ```
- **`GPTQ/env.yml`**：
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
      - gptqmodel>=7.0          # AutoGPTQ 继任，pip pre-built wheel
      - sentencepiece
      - lm-eval>=0.4.2          # 跑量化后 PPL + zero-shot
      # 其它依赖（transformers / accelerate / datasets）跟着 gptqmodel 自动装
  ```
- **没有 examples/**：GPTQModel 是 library，没有像 AutoGPTQ 那种 `examples/quantization/quant_with_alpaca.py`。Phase 1 入口写一个 `GPTQ/quant_eval.py`，**~10 行原样 copy 自 GPTQModel README quickstart**：

  ```python
  # GPTQ/quant_eval.py
  """Phase 1 GPTQ — verbatim copy of GPTQModel README quickstart, with model_id swappable."""
  import sys
  from datasets import load_dataset
  from gptqmodel import GPTQConfig, GPTQModel

  CANONICAL = "--canonical" in sys.argv
  model_id   = "meta-llama/Llama-2-7b-hf" if CANONICAL else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  quant_path = "results/quantized_w4g128" if CANONICAL else "results/smoke/quantized"
  n_calib    = 1024 if CANONICAL else 128

  calibration_dataset = load_dataset(
      "allenai/c4",
      data_files="en/c4-train.00001-of-01024.json.gz",
      split="train",
  ).select(range(n_calib))["text"]

  quant_config = GPTQConfig(bits=4, group_size=128)
  model = GPTQModel.load(model_id, quant_config)
  model.quantize(calibration_dataset, batch_size=1)
  model.save(quant_path)
  print(f"Done. Quantized model saved to {quant_path}")
  ```

  > 这是 Phase 1 GPTQ 唯一一份 Python 代码，每行都来自上游 README。**不算手搓**——是上游推荐的最小用法。
- **跑法**：
  ```bash
  # 烟雾跑（本地 12GB / TinyLlama）
  cd GPTQ
  python quant_eval.py 2>&1 | tee results/smoke/stdout.txt

  # Canonical（lab 96GB / LLaMA-2-7B）
  python quant_eval.py --canonical 2>&1 | tee results/canonical_w4g128_quant_stdout.txt

  # 评测（lm-eval 一站式 PPL + zero-shot）
  python -m lm_eval \
      --model hf \
      --model_args pretrained=results/quantized_w4g128 \
      --tasks wikitext,piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
      --output_path results/canonical_w4g128_eval.json
  ```
- **论文 anchor**: LLaMA-2-7B / W4-g128 / WT2 PPL ≈ 5.69（baseline 5.47，容差 ±0.3）。
  > **注**：GPTQModel 是 AutoGPTQ 的 fork/refactor，底层算法仍是 GPTQ 但实现细节有差异（更快 / 更省内存）；实测可能与 AutoGPTQ 时代 5.69 anchor 略漂，落 ±0.3 内即可。
- **跑时间**：30–60 min on lab 96GB。
- **Phase 2 研读**：读 `third_party/GPTQModel/gptqmodel/quantization/gptq.py`；推荐对照 archive 的 AutoGPTQ 同名文件 diff，看 fork 改了什么 —— 哪些是 OBQ 论文核心、哪些是工程优化。这本身就是 Phase 2 笔记的好素材。

### §6.3 KIVI（流程最不同）

- **不需要 calibration**。KIVI 是 inference-time KV cache 量化（per-channel keys + per-token values, 2bit），跟权重量化是两个层面。
- **vendor + 编译 CUDA extension**（必须本机有 CUDA Toolkit + nvcc，§1.3）：
  ```bash
  cd KIVI
  git submodule add https://github.com/jy-yuan/KIVI third_party/KIVI
  ```
- **`KIVI/env.yml`**：
  ```yaml
  name: quant-kivi
  channels:
    - pytorch
    - nvidia
    - conda-forge
  dependencies:
    - python=3.10
    - pip
    - pytorch=2.1.*            # KIVI 官方 README 指定 2.1.x
    - pytorch-cuda=12.1        # 必须与下面 nvcc 12.1 一致
    - cuda-toolkit=12.1        # 提供 nvcc 用于编译自定义 kernel（lab 服务器上可走 module load 替代）
    - pip:
      - -r requirements.txt
  ```
- **vendor 后装依赖 + 编译 CUDA kernel**：
  ```bash
  conda activate quant-kivi
  cd ~/quant-reproduce/KIVI/third_party/KIVI
  pip install -r requirements.txt                  # 装 transformers / datasets / flash-attn 等
  cd quant && python setup_cuda.py install         # ← 编译，~5-10 分钟
  ```
  - **常见编译失败**：nvcc 版本与 PyTorch 自带 CUDA runtime 不匹配。检查：
    ```bash
    nvcc --version          # 编译器
    python -c "import torch; print(torch.version.cuda)"   # PyTorch 期望
    ```
    两者主版本号要一致（都是 12.1 或都是 11.8）。conda env 里 `pytorch-cuda=12.1` + `cuda-toolkit=12.1` 同 channel 装就一致。
- **上游复现脚本**：KIVI 的脚本通常在 `third_party/KIVI/` 根目录或 `eval/`：
  - `pred.py` / `example_chat.py` — 推理 demo
  - `eval/ppl_eval.py` — PPL 评测（具体名以 vendor 时为准）
- **跑法**：
  ```bash
  cd KIVI/third_party/KIVI
  python eval/ppl_eval.py \
      --model_name meta-llama/Llama-2-7b-hf \
      --k_bits 2 --v_bits 2 --k_group_size 32 --v_group_size 32 \
      --use_kivi True \
      2>&1 | tee ../../results/canonical_kivi2_stdout.txt
  ```
  > 上游脚本在 KIVI repo 里也叫各种名字，**以 vendor 时实际看到的为准**。KIVI 的 README 通常给一个 "Reproducing the paper" 段落。
- **KIVI 跑法的本质差异**：
  - **不需要** calibration（KIVI 是 inference-time KV cache 量化）
  - **不需要** 量化后保存 checkpoint（KIVI monkey-patch 模型的 attention 层，没有"量化产物"这个概念）
  - 所以脚本就是"加载模型 → 打补丁 → 跑 PPL"一步到位
- **关键评测差异**：还要测 KV cache 显存（不只权重）。上游脚本通常会自带；如果没有，自己 `nvidia-smi` 在跑 long-context PPL 时记一下峰值。Phase 3 的 `common/eval/memory.py` 会补 `measure_kv_cache_bytes`。
- **论文 anchor**: LLaMA-2-7B + KV-2bit / WT2 PPL 几乎不掉（容差 ±0.3）。

---

## §7 数字打不到目标的排查流程

按顺序排查（所有方法通用）：

### Step 1 — 模型 ID 与 HF commit SHA

打开 `meta_<config>.json` 看 `model` 字段，对照 HuggingFace 模型卡的当前 commit。HF 偶尔会换权重。LLaMA-2-7B **必须用** `meta-llama/Llama-2-7b-hf`（不是 NousResearch 等镜像，那个 tokenizer 略有差异）。

### Step 2 — Calibration 切片

确认你跑的命令里 calibration 数量、seq_len 跟论文一致。换 seed 重跑一次看波动多大：

```bash
# 修改上游 example 的 --seed / --num_samples 参数
python examples/quantize.py --seed 7 ...   # 以方法的 example 实际参数为准
```

差异 < 0.1 PPL = 正常波动；差异 > 0.5 PPL = calibration 路径有问题（数据集版本？切片逻辑？）。

### Step 3 — 官方 repo 的 issue 区

```
https://github.com/casper-hansen/AutoAWQ/issues?q=reproduce
https://github.com/Aaronhuang-778/BiLLM/issues?q=reproduce
https://github.com/jy-yuan/KIVI/issues?q=reproduce
https://github.com/ModelCloud/GPTQModel/issues?q=reproduce
https://github.com/AutoGPTQ/AutoGPTQ/issues?q=reproduce  # archive 但历史有用
```

搜 `"reproduce"` / `"PPL difference"`，看作者怎么回。

### Step 4 — 检查上游版本

看你 vendor 的 commit。`cd third_party/<repo>; git log -1` 对照论文发表时间。上游后续 commit 改算法核心（比如 GPTQ 的 Cholesky 数值稳定性 patch、GPTQModel fork 自 AutoGPTQ 后的微调），数字都可能漂。

```bash
# 退到论文当时的 commit 看看
cd third_party/<UPSTREAM>
git log --oneline | head -20
git checkout <某个 tag 或 SHA>   # 论文发表前后的 commit
```

### Step 5 — 如实记录

差距找不到原因 → 在子目录 `README.md` 末尾加：

```markdown
## 实测异常记录

| 配置 | 实测 PPL | 论文 anchor | 差距 | 已排查步骤 |
|------|---------|------------|------|-----------|
| w4g128 | 6.21 | 5.69 | +0.52 | 1.✓ HF SHA 对得上 2.✓ seed/calib/seq 都按协议 3.✗ 官方 issue 暂无相同问题 4.✓ 官方 eval 脚本同环境跑也得 6.18 |

**结论**：差距与官方实现自身一致。可能是 GPTQModel（AutoGPTQ 继任）与原 GPTQ 论文实现的 Cholesky 数值差异，或 fork 后某次 refactor。可在 Phase 2 笔记里展开。
```

> **不要为了好看刷参数**。作品集真正可信度在这里 —— 申请委员会更看重你"诚实地排查并记录"的能力，而不是拼凑出与论文严格一致的数字。

---

## §8 完成判定

### 单方法完成（Phase 1 + Phase 2 of 该方法）

`✅` 所有条目都满足：

- [ ] `<METHOD>/results/results_<config>.json` 存在，含 `ppl_wikitext2` + `zeroshot` (6 项) + `memory`
- [ ] `<METHOD>/results/meta_<config>.json` 含完整元数据
- [ ] `<METHOD>/README.md` "实测 vs 论文" 表填好（数字达 §3.4 容差，或写明 §7 的排查记录）
- [ ] `docs/reports/<method>.md` 五节都写完（不是模板）
- [ ] git log 干净（每个里程碑一个 commit）
- [ ] 打 tag `phase1-<method>-done`

### Plan A (Phase 1+2 of GPTQ) 完成

GPTQ 单方法完成 + 顶层 README 把 GPTQ 标 ✅ + `docs/results/summary.md` 有 GPTQ 一行。

### Phase 1+2 全部完成（四方法都 ✅ 后）

- [ ] 顶层 `README.md` 4 个方法都 ✅
- [ ] `docs/results/summary.md` 大表 4 行
- [ ] git tag `phase1-all-done`
- [ ] **此时才**写 Phase 3 spec：`docs/superpowers/specs/YYYY-MM-DD-phase3-unified-pipeline-design.md`

---

## §9 Colab 备用通道

什么时候用：
- Lab 服务器排队 / 暂时不可用
- 本地 12GB 想跑稍大的模型（TinyLlama 烟雾过了，想顺手试 LLaMA-2-7B 但本地 12GB OOM）

### §9.1 在 Colab 跑 AWQ canonical（示例；其它方法换 vendor URL + 上游脚本即可）

新建 Colab notebook，选 **A100 GPU runtime**（Pro 才有）：

```python
# Cell 1: clone 仓库（含 submodule）
!git clone --recurse-submodules https://github.com/你的账号/quant-reproduce.git
%cd quant-reproduce

# Cell 2: 装上游依赖（Colab 自带 CUDA torch，不要再装 torch）
!pip install autoawq>=0.2 autoawq-kernels sentencepiece lm-eval>=0.4.2

# Cell 3: HF login
from huggingface_hub import login
login("hf_xxxxxxxx")  # 你的 token

# Cell 4: 预下载 LLaMA-2-7B（Colab 默认 HF cache 在 /root/.cache/huggingface）
import os
os.environ["HF_HOME"] = "/content/hf_cache"
from transformers import AutoModelForCausalLM
AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Cell 5: 跑上游 example（AWQ 量化）
%cd AWQ/third_party/AutoAWQ
!python examples/quantize.py \
    --model_path meta-llama/Llama-2-7b-hf \
    --quant_path ../../results/quantized_w4g128 \
    --w_bit 4 --q_group_size 128 \
    2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt

# Cell 6: 跑评测（lm-eval-harness 一站式：PPL + zero-shot）
!python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128 \
    --tasks wikitext,piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --output_path ../../results/canonical_w4g128_eval.json

# Cell 7: 把 results 下载到本地（保险措施，Colab session 会断）
from google.colab import files
files.download("../../results/canonical_w4g128_quant_stdout.txt")
files.download("../../results/canonical_w4g128_eval.json")
```

> 跑 GPTQ 时把 Cell 2 的 `autoawq` 改成 `gptqmodel`，Cell 5 替换为 `python ../../quant_eval.py --canonical`（你 Phase 1 GPTQ 子目录里的 quant_eval.py）。

### §9.2 注意事项

- **Colab session 限时**：免费 12 小时、Pro 24 小时，断后丢失 cache。每次跑长任务**及时下载** results JSON。
- **挂载 Google Drive 持久化 cache**：
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"   # 14GB 模型存到 Drive
  ```
- **Colab 的 GPU 型号会浮动**：T4 / V100 / A100 / L4。LLaMA-2-7B canonical 至少要 V100 16GB；A100 40GB 最稳。
- **Colab 不算"主路径"**：拿到的数字进 git 时，`meta.json` 会记录是 Colab 跑的（GPU name = "Tesla A100-SXM4-40GB" 等），README 里如实写"在 A100 / Colab Pro 复跑"。

---

## §10 常见坑速查

| # | 现象 | 原因 | 解决 / 提前预防 |
|---|------|------|----------------|
| 1 | `pip install gptqmodel` / `autoawq` 报 `torch not found` | conda env 没装 CUDA torch | 先 `pip install torch --index-url https://download.pytorch.org/whl/cu121` 再装方法包；或在 `env.yml` pin `pytorch-cuda=12.1` |
| 2 | `torch.cuda.is_available()` 返回 False | 装了 CPU 版 torch | 卸掉重装：`pip uninstall torch -y; pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 3 | `.\scripts\env_local.ps1` 报 `running scripts is disabled` | PS 执行策略默认禁 | 一次性：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 4 | `huggingface_hub` 报 `no symlinks` warning（Windows） | Windows 默认不开发者模式 | 不影响功能，warning 可忽略；或 Settings → Developer Mode 开 |
| 5 | LLaMA-2 下载 401/403 | 没授权 / 没登录 | `huggingface-cli login` + 同意 license 页面（§2.2） |
| 6 | `conda activate` 在 .sh 里失败 | 非交互 shell 没加载 conda 函数 | 我们的 `run_phase1_method.sh` 已 `source conda.sh`；自定义脚本要照做 |
| 7 | Lab 上 git push 被拒 | 默认分支保护或缺 git 身份 | `git config --global user.email ...` + `git config --global user.name ...` |
| 8 | 上游 PPL 脚本输出 NaN | fp16 在长 seq 数值溢出 | 看上游脚本是否有 `--dtype fp32` 选项；或 issue 区搜"NaN" |
| 9 | BiLLM 跑到一半 OOM | 二值化中间张量大 | 减 batch；canonical 必须用 96GB lab，不要在 12GB 本地跑 |
| 10 | KIVI CUDA 编译失败 | nvcc 与 PyTorch CUDA 版不匹配 | `nvcc --version` 与 `python -c "import torch; print(torch.version.cuda)"` 主版本号要一致 |
| 11 | `pytest --collect-only` 报 `Unknown mark` | 标记没在 pyproject 注册 | 已在 pyproject.toml 注册 `slow` 和 `require_cuda`；自定义新标记要追加 |
| 12 | `pip install -e .` 在 Windows 报路径太长 | Windows 默认 MAX_PATH 260 | 注册表开 LongPathsEnabled，或装到短路径（D:\Q\repro） |
| 13 | `lm-eval-harness` 第一次跑卡很久不动 | 在下 dataset（PIQA ~10MB） | 等 / 设代理；走完一次后 cache 在 `~/.cache/huggingface/datasets` |
| 14 | git submodule 没拉下来上游代码 | clone 时漏了 `--recurse-submodules` | `git submodule update --init --recursive` |
| 15 | git push 出现 `LF will be replaced by CRLF` warning | Windows 默认 `core.autocrlf=true` | 不影响（git 内部存 LF，Linux 拉下来正常）；Warning 可忽略 |
| 16 | Colab session 断了 results 没保存 | runtime 超时 | 下次记得 §9.1 最后一个 Cell `files.download()` 取下来；或挂 Drive 持久化 |
| 17 | 上游 example 参数名跟我文档对不上 | 上游版本变了 | 用 `--help` 看实际参数；上游脚本 commit history 看改动 |

---

## §11 接下来你具体做什么

1. **现在**（已经完成）：framework 全部就绪，所有 `common/` 测试通过。**`common/` 暂时不动**，留给 Phase 3。
2. **下一步**：按 §5 走 **AWQ** 完整流程（顺序修订 2026-05-10：AWQ 第一）：vendor → 建 env → 跑上游 `examples/quantize.py`。**不写自己的 Python 代码**。
3. **第一里程碑**：跑出 AWQ canonical 数字（PPL ≈ 5.60 ± 0.3） → tag `phase1-awq-done`。
4. **告诉我数字** → 我写 Plan B (BiLLM)。BiLLM / KIVI / GPTQ 都按 §5 模板，差异在 §6。

AWQ 做完之后的预期 commit 历史：

```
* (tag: phase1-awq-done) docs: link AWQ as completed + initialize summary table
* docs(reports): add Phase 2 source-code study for AWQ          ← 可选，可推迟到全部方法都做完
* docs(awq): fill in canonical numbers vs paper anchor + meta   ← 抽 eval.json 数字写表
* data(awq): canonical W4-g128 results on LLaMA-2-7B            ← 上游 stdout + eval.json
* test(awq): smoke run with TinyLlama passes (results gitignored)
* scaffold(awq): vendor AutoAWQ + env.yml + README template
```

只有 **2 个 commit 需要写代码**（scaffold = 写 env.yml + README + .gitmodules；docs = 填表+元数据）。其他都是跑完上游脚本 + commit 产物。

**全部 4 方法的预期里程碑顺序**：

```
phase1-awq-done  →  phase1-billm-done  →  phase1-kivi-done  →  phase1-gptq-done  →  phase1-all-done
```

> Phase 3 才是真正"手搓"的阶段：那时你写自己的 unified pipeline，把 4 方法纳入同一 API，用 `common/` 做共享 eval。Phase 1 / 2 不沾。
>
> Plan A 是 GPTQ 那一份（最后做）；AWQ / BiLLM / KIVI 等你 ping 我后单独写。或者你也可以照本文档 §5 直接开始 AWQ，不一定要等 plan。
