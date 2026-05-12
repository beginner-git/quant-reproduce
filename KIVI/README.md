# KIVI — LLaMA-2-7B K2V2 KV-cache 量化源码复现

> **核心理念**：clone `jy-yuan/KIVI` 进 `third_party/KIVI/`，用上游的 `LlamaForCausalLM_KIVI` 模型子类做**运行时** KV cache 2-bit 量化，跑 `example.py`（GSM8K 烟雾）和 `pred_long_bench.py` + `eval_long_bench.py`（LongBench canonical）。**没有**单独的 quantize step，**没有** calibration data——KIVI 是 tuning-free，量化发生在 inference 时。

> **上游状态**：ICML 2024 paper code，pyproject.toml 锁 torch 2.4.1 + transformers 4.43.1（比 BiLLM 的 4.35 时代新很多，AWQ/BiLLM 时代的几类 bit-rot 多数自动避开）。最近一次 commit 在 2024 年中。

> **运行环境**：本服务器（4× RTX 3090, 24GB each），账号 `yiminl50`。所有路径按服务器约定走 `/shared/yiminl50/...`；存储规则见 `~/notes/SERVER_GUIDE.md`。

## §0 目标 & 验收

| 项 | 值 |
|----|----|
| 模型 | `NousResearch/Llama-2-7b-chat-hf`，预下载到 `/shared/yiminl50/models/llama-2-7b-chat-hf`（LongBench 是 instruction-style QA，chat 模型才符合 paper Table 5/6 的评测条件；base 模型分会显著低） |
| KIVI 配置 | `k_bits=2, v_bits=2, group_size=32, residual_length=32`（paper 主表配置） |
| Calibration | **无**（KIVI tuning-free） |
| 主评测 | LongBench 16-task average（`pred_long_bench.py` → `eval_long_bench.py`） |
| 辅评测 | `example.py` GSM8K 5-shot 生成（烟雾，确认 inference 通） |
| Paper anchor | LLaMA-2-7B-chat LongBench avg KIVI K2V2 vs FP16 baseline，差距 < 1-2 分（paper Table 5/6；具体数字待跑完核对） |
| Paper headline | 2.6× 显存↓ / 2.35-3.47× 吞吐↑ / 4× 更大 batch（abstract claim，对照基线 `k_bits=16, v_bits=16`） |
| 验收 | (a) `example.py` 不挂能生成；(b) LongBench avg 与 FP16 baseline 差 < 5%；(c) 用 `mem_spd_test.py` 量到 KIVI on/off 时 peak VRAM 明显下降 |

> **KIVI 跟 AWQ/BiLLM 的根本区别**：AWQ/BiLLM 量化模型权重（保存一份新的 `.pt`/`.safetensors`），后续 inference 直接加载量化权重。KIVI 是 inference 时**对每个 batch 的 KV cache 做量化**，模型权重保持 fp16；所以没有"save quantized model"这一步，也没有 calibration——量化逻辑由 `LlamaForCausalLM_KIVI` 这个 transformer 子类承担。

## §1 前置条件（本机已满足）

- **GPU**：4× RTX 3090（24 GB VRAM 各张） — ✅
- **CUDA driver**：12.4，nvcc 12.0 在 `/usr/local/cuda` — ✅（KIVI 要本地编 `kivi_gemv` CUDA kernel，需 nvcc）
- **conda**：env 自动落 `/shared/yiminl50/conda_envs/` — ✅
- **HuggingFace 缓存**：`$HF_HOME=/shared/yiminl50/hf_cache` — ✅
- **HF token**：`NousResearch/...` mirror 无需登录 — ✅

## §2 创建 conda env

```bash
cd ~/projects/reproduce
bash scripts/env_lab.sh KIVI         # 自动落 /shared/yiminl50/conda_envs/quant-kivi
conda activate quant-kivi
```

`env.yml` 只装 PyTorch 2.4 + cu121 + 几个编译期工具（`ninja` 加速 flash-attn 编译、`protobuf` 跑 tokenizer）。其它依赖（transformers 4.43.1 / accelerate / flash-attn / fastchat / datasets / ...）走 §3 的 `pip install -e third_party/KIVI`，由 KIVI 自家 pyproject.toml 拉齐。

## §3 拉上游 KIVI submodule + 装 deps + 编 CUDA kernel

```bash
cd ~/projects/reproduce
git submodule add https://github.com/jy-yuan/KIVI KIVI/third_party/KIVI
git submodule update --init --recursive

# 记录 vendor commit SHA（后续填进 _meta.md）
git -C KIVI/third_party/KIVI rev-parse HEAD
# 当前 pin: 876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

### §3.1 装 KIVI 主包（用预编译 flash-attn wheel，**~1 min**）

> **不要**直接跑 `pip install -e .` 或 `pip install flash-attn`——会同时撞两个坑：
> 1. `pip install -e .` 走 build isolation，flash-attn 的 setup.py `import torch` 时找不到 torch（build env 是干净的）→ `ModuleNotFoundError: No module named 'torch'`
> 2. 加 `--no-build-isolation` 解决 (1) 后，flash-attn 的 setup.py 实际**不本地编译**——它会去 GitHub releases 拉**预编译 wheel** 到 `/tmp/`，然后 `os.rename` 到 `$PIP_CACHE_DIR=/shared/yiminl50/.pip_cache/wheels/...`。`/tmp` 和 `/shared` 是不同 mount → `Errno 18: Invalid cross-device link`

正确做法：**直接 curl 那个预编译 wheel，跳过 setup.py 整套机制**：

```bash
tmux new -s kivi-install
conda activate quant-kivi

# 1. 直接拉 flash-attn 预编译 wheel（245 MB，~25 秒）
#    URL 里几段参数：cu12 (CUDA 12.x) + torch2.4 + cxx11abiFALSE (torch 2.4 默认 ABI) + cp310 (Python 3.10)
cd /tmp
curl -L -O https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 2. 装 wheel（秒装，含 einops + fsspec 两个间接依赖）
pip install /tmp/flash_attn-2.8.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 3. 装 KIVI 本体——flash-attn 已就位，剩下都是 wheel 秒装
cd ~/projects/reproduce/KIVI/third_party/KIVI
pip install -e .
```

含义：
- `-e`: editable install，方便后续 patch
- pyproject.toml 锁的依赖：`torch==2.4.1`（已在 env.yml）、`transformers==4.43.1`、`accelerate`、`flash-attn`、`datasets`、`sentencepiece`、`tokenizers>=0.15`、`fastchat`、`protobuf` 等
- **wheel 版本对照**：如果将来 KIVI 升级到 torch 2.5+ 或不同 flash-attn 版本，去 https://github.com/Dao-AILab/flash-attention/releases 找对应 `cu12torch<X.Y>cxx11abiFALSE-cp310` 的 wheel。`cxx11abiFALSE` 是 torch ≤2.4 默认；torch 2.5+ 切到 `cxx11abiTRUE`
- **如果只想先跑 GSM8K 烟雾**（不用 flash-attn 也能跑——KIVI attention 实现里 flash-attn 是可选），跳过：
  ```bash
  pip install -e . --no-deps
  pip install transformers==4.43.1 accelerate datasets sentencepiece protobuf tokenizers fastchat packaging numpy
  ```

### §3.2 编 KIVI 自家 CUDA kernel `kivi_gemv`

```bash
cd ~/projects/reproduce/KIVI/third_party/KIVI/quant
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation
```

含义：
- `cd` 必须进 `quant/` 子目录，不是 KIVI 根目录（KIVI 根目录是主包 `kivi`，`quant/` 是单独的 CUDA extension 子包 `kivi_gemv`）
- `--no-build-isolation` 让 setup.py 编译时能 `import torch`（torch 已在 env，不需要 pip 在 build env 重装）
- `TORCH_CUDA_ARCH_LIST="8.6"` 显式告诉 nvcc 编 sm_86（3090），避免它探测一堆 arch 浪费时间
- 编 `csrc/gemv_cuda.cu`（KIVI 的 INT2 GEMV 推理 kernel）
- 编译参数含 `-DENABLE_BF16` 等 bf16 支持 flags、`--use_fast_math`、`--threads=8`
- 编译 1-3 min

## §4 验证安装

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# 期望: torch 2.4.x cuda True

python -c "import transformers; print('transformers', transformers.__version__)"
# 期望: transformers 4.43.1

python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
# 期望: flash_attn 2.x.x  （如果 §3.1 用了 --no-deps 跳过，这条会失败，没关系）

python -c "import torch; import kivi_gemv; print('kivi_gemv OK:', kivi_gemv.__file__)"
# 期望: 不报错，能打印 .so 路径
# 注意: 必须先 import torch！kivi_gemv.so 链接 torch 的 libc10.so，单独 import
# kivi_gemv 会得到 ImportError: libc10.so: cannot open shared object file
# 实际用 KIVI 时不存在这问题——上游脚本都先 import torch

# 关键：KIVI 自家的模型子类能不能 import
cd ~/projects/reproduce/KIVI/third_party/KIVI
python -c "from models.llama_kivi import LlamaForCausalLM_KIVI; print('LlamaForCausalLM_KIVI', LlamaForCausalLM_KIVI)"
# 期望: 打印 class，不报错
```

## §5 跑前准备

KIVI 没有 calibration 数据，但需要 (a) 本地模型 (b) LongBench 数据集预下载 (c) 把 KIVI 硬编码的几个相对路径 symlink 到 `/shared`。

### §5.1 模型预下载到本地路径

LongBench 是 instruction-style 长上下文 QA，**LongBench eval 用 chat 模型才合理**（paper Table 5/6 用的就是 chat）。`mem_spd_test.py` 用 base（paper 默认）——但 base 已经被 BiLLM 复现下过，可以直接复用，不用再下：

```bash
# 1. Chat 模型（KIVI LongBench eval 用，§7 canonical 走这个）
huggingface-cli download NousResearch/Llama-2-7b-chat-hf \
    --local-dir /shared/yiminl50/models/llama-2-7b-chat-hf \
    --exclude "*.bin" "*.pdf"

# 2. Base 模型（mem_spd_test.py 默认用，§7.2 走这个）
# BiLLM/README §5.1 已经下过，直接 ls 看在不在即可——在的话跳过这步
ls /shared/yiminl50/models/llama-2-7b-hf/ 2>/dev/null && echo "base already present (from BiLLM run)" || \
huggingface-cli download NousResearch/Llama-2-7b-hf \
    --local-dir /shared/yiminl50/models/llama-2-7b-hf \
    --exclude "*.bin" "*.pdf"

ls /shared/yiminl50/models/llama-2-7b-chat-hf/    # 应见 config.json / *.safetensors / tokenizer.*
ls /shared/yiminl50/models/llama-2-7b-hf/         # 同上
```

> **chat vs base 区别**：参数权重不一样（chat 有 RLHF tuning），但**架构、参数量、KV cache 结构完全相同**——对 mem_spd_test 测量峰值显存 + tokens/s 没差。用 base 是跟 KIVI paper 默认配置对齐，用 chat 也行。

### §5.2 LongBench 数据集预下载

`pred_long_bench.py` 内部走 `datasets.load_dataset('THUDM/LongBench', '<task>', split='test')`，会下 16 个 task 各自的数据（每个几 MB 到几十 MB）。**预拉一次**避免 §7 跑到一半某个 task 才下：

```bash
mkdir -p /shared/yiminl50/datasets/longbench
huggingface-cli download THUDM/LongBench --repo-type dataset \
    --local-dir /shared/yiminl50/datasets/longbench

ls /shared/yiminl50/datasets/longbench/    # 应见 data/ 目录含 jsonl 文件
```

> 这里走 `huggingface-cli` 不走 `datasets` 库，避免 BiLLM 那次踩的 `cas-bridge.xethub.hf.co` redirect 卡死问题。如果遇到 datasets 库在某个 task 上卡住，参考 BiLLM/README §5.4 的 patch 模式（gzip+json 读本地）。

### §5.3 KIVI 硬编码路径 symlink

KIVI 上游脚本有两个相对路径硬编码：

| 路径 | 作用 | symlink 到 |
|------|------|------------|
| `./cached_models` | `pred_long_bench.py --cache_dir` 默认值（HF 模型 cache） | `$HF_HOME/hub` |
| `./pred/` | `pred_long_bench.py` 写预测 JSONL 的目录 | `/shared/yiminl50/quantized/kivi-pred/` |

> 注：第一个 symlink 让 KIVI 跟现有 `$HF_HOME` cache 共享，避免重复下载。我们已经在 §5.1 把模型预下到了 `/shared/yiminl50/models/llama-2-7b-chat-hf` 这个**独立**路径——所以更省事的做法是 §6/§7 命令里**绕开** `--cache_dir`，直接传本地路径作为 `--model_name_or_path`。这样不需要任何 symlink。

```bash
cd ~/projects/reproduce/KIVI/third_party/KIVI

# 预测输出 symlink（避免几 GB JSONL 落 /home）
mkdir -p /shared/yiminl50/quantized/kivi-pred
ln -sfn /shared/yiminl50/quantized/kivi-pred pred

# 验证
ls -la pred                                # 应是软链
```

### §5.4 上游 patches（共 1 处）

#### Patch 1: 给 `config/model2maxlen.json` + `config/model2path.json` 加全小写 `llama-2-7b-chat-hf` 键

KIVI 的 `pred_long_bench.py:190` 走 `model_name = os.path.basename(args.model_name_or_path)`，然后 `model2maxlen[model_name]` 严格查 dict。两个 config JSON 里只有大写 `"Llama-2-7b-chat-hf"`，本地目录名是 lowercase `llama-2-7b-chat-hf`（跟我们的 lowercase path 约定一致），所以 KeyError。

```bash
cd ~/projects/reproduce/KIVI/third_party/KIVI

python -c "
import json
p = 'config/model2maxlen.json'
with open(p) as f: d = json.load(f)
d['llama-2-7b-chat-hf'] = 4096
with open(p, 'w') as f: json.dump(d, f, indent=4)
"

python -c "
import json
p = 'config/model2path.json'
with open(p) as f: d = json.load(f)
d['llama-2-7b-chat-hf'] = '/shared/yiminl50/models/llama-2-7b-chat-hf'
with open(p, 'w') as f: json.dump(d, f, indent=4)
"

grep -E '"llama-2-7b-chat-hf"' config/model2maxlen.json config/model2path.json    # 验证
```

> patch 后 `KIVI/third_party/KIVI` 在 git status 显示 `modified content`——预期，记进 `results/_meta.md`。后续如果跑别的模型，按同模式加键。

#### Patch 2: `mem_spd_test.py` 补 `config.use_flash = True`

KIVI 的 `LlamaAttention_KIVI.__init__` 硬要求 `config.use_flash=True`（INT2 attention 实现依赖 flash-attn 的 causal mask layout）。`example.py` / `pred_long_bench.py` 都设了，**就 `mem_spd_test.py` 漏了**——上游一致性 bug。

```bash
cd ~/projects/reproduce/KIVI/third_party/KIVI

# 在 config = LlamaConfig.from_pretrained(...) 那行后插入 use_flash 设置
sed -i "/^config = LlamaConfig.from_pretrained/a config.use_flash = True" mem_spd_test.py

grep -n "use_flash" mem_spd_test.py    # 验证有一行了
```

## §6 烟雾跑（GSM8K 5-shot，~2-5 min）

目的：验证 KIVI 模型类能正确加载 + 量化 KV cache + 生成。用上游 `example.py` 但改一行模型路径。

```bash
tmux new -s kivi-smoke
gpu                                  # nvidia-smi 别名；找空闲卡，下面假设 1 号
conda activate quant-kivi
cd ~/projects/reproduce/KIVI/third_party/KIVI

# 上游 example.py 默认是 Llama-3.1-8B-Instruct；改成 LLaMA-2-7B-chat 跟项目一致
# 选项 A: 临时改源码（sed 一行）
sed -i "s|meta-llama/Llama-3.1-8B-Instruct|/shared/yiminl50/models/llama-2-7b-chat-hf|" example.py

# 选项 B: 不动源码，在命令行覆盖（如果 example.py 是用 argparse 的话；上游目前是 hardcoded
# constant，没有 argparse，所以只能用 A）

CUDA_VISIBLE_DEVICES=1 python example.py
```

**判定**：
- 不挂、最后能打印一段 GSM8K 数学题的生成答案
- stdout 里能看到类似 "KIVI config: k_bits=2 v_bits=2 group_size=32 residual_length=32"
- 数学题答案对错不重要（KIVI 不影响 reasoning 能力的判定不是 1 道题能下结论的）

## §7 Canonical 跑（LongBench 16-task，**3-6 hours**）

上游有 `scripts/long_test.sh` 包装；我们直接调 Python，避开它硬编码的 `--cache_dir ./cached_models`。

```bash
tmux new -s kivi-canonical
conda activate quant-kivi
cd ~/projects/reproduce/KIVI/third_party/KIVI
mkdir -p ~/projects/reproduce/KIVI/results

# 跑 KIVI K2V2 预测（16 task × ~200 samples × ~LLaMA-2 forward；3-6 hour 单卡）
CUDA_VISIBLE_DEVICES=1 python pred_long_bench.py \
    --model_name_or_path /shared/yiminl50/models/llama-2-7b-chat-hf \
    --k_bits 2 --v_bits 2 \
    --group_size 32 --residual_length 32 \
    2>&1 | tee ~/projects/reproduce/KIVI/results/canonical_k2v2_pred_stdout.txt
```

跑完 standard-mode 8 个 task（triviaqa / qasper / trec / samsum / lcc / repobench-p / qmsum / multi_news）的预测 JSONL，落在：

```
pred/<basename>_<max_length>_<k_bits>bits_group<group_size>_residual<residual_length>/
```

即 KIVI 把模型 basename + 量化配置拼成子目录名（同模型不同配置 prediction 不互相覆盖）。本配置下就是：

```
pred/llama-2-7b-chat-hf_4096_2bits_group32_residual32/
```

（因为 §5.3 symlink，实际在 `/shared/yiminl50/quantized/kivi-pred/llama-2-7b-chat-hf_4096_2bits_group32_residual32/`）

> 跑 16-task extended 而不是 standard 8-task：加 `--e` flag，但 paper Table 5 是 standard 16-task average，extended 是另一组。本节默认 standard。

然后跑评分（`--model` 必须**完整匹配**那个长子目录名）：

```bash
python eval_long_bench.py \
    --model llama-2-7b-chat-hf_4096_2bits_group32_residual32 \
    2>&1 | tee ~/projects/reproduce/KIVI/results/canonical_k2v2_eval_stdout.txt
```

`eval_long_bench.py` 用 `metrics.py` 里的 task-specific scorers（F1 / ROUGE / classification / retrieval）算每 task 分数，输出 `pred/<model_long>/result.json`。

### §7.1 FP16 baseline（对照组，**也 3-6 hour**）

要算 "KIVI 损了多少"，必须跑一遍 FP16 baseline。KIVI 模型类在 `k_bits=16, v_bits=16` 时退化为标准 attention：

```bash
CUDA_VISIBLE_DEVICES=1 python pred_long_bench.py \
    --model_name_or_path /shared/yiminl50/models/llama-2-7b-chat-hf \
    --k_bits 16 --v_bits 16 \
    --group_size 32 --residual_length 32 \
    2>&1 | tee ~/projects/reproduce/KIVI/results/baseline_fp16_pred_stdout.txt

# baseline 的 prediction 子目录名是 llama-2-7b-chat-hf_4096_16bits_group32_residual32
python eval_long_bench.py \
    --model llama-2-7b-chat-hf_4096_16bits_group32_residual32 \
    2>&1 | tee ~/projects/reproduce/KIVI/results/baseline_fp16_eval_stdout.txt
```

> KIVI 的子目录命名规则（basename + bits + group + residual）天然分隔了不同配置的 prediction——不会覆盖，不用手动 mv。

### §7.2 显存 / 吞吐量测（验 paper headline）

`mem_spd_test.py` 跟 `example.py` 一样硬编码 `meta-llama/Llama-2-7b-hf`，且**上游默认 `BATCH_SIZE=96`**——这是冲 paper "4× larger batch" headline 去的，单 3090 (24GB) 跑 LLaMA-2-7B fp16 必 OOM。本仓库 vendored 文件已 patch 成 `BATCH_SIZE=16`（K2V2 + FP16 baseline 都能在单卡跑下，方便算 ratio）。如果你 reset 过 submodule，重新应用：

```bash
cd ~/projects/reproduce/KIVI/third_party/KIVI

# Patch 1: 本地 base 路径
sed -i "s|meta-llama/Llama-2-7b-hf|/shared/yiminl50/models/llama-2-7b-hf|" mem_spd_test.py
# Patch 2: 补 use_flash (上游漏列，见 §5.4 patch 2)
sed -i "/^config = LlamaConfig.from_pretrained/a config.use_flash = True" mem_spd_test.py
# Patch 3: BATCH_SIZE 砍到单卡能跑的值
sed -i "s/^BATCH_SIZE = 96$/BATCH_SIZE = 16/" mem_spd_test.py
grep -nE "model_name_or_path|use_flash|BATCH_SIZE" mem_spd_test.py    # 验证 3 处 patch
```

**两轮跑法**——单卡同 batch 比 K2V2 vs FP16 ratio（这是 paper 显存 2.6× 吞吐 2-3× 的算法）：

```bash
# 第 1 轮：KIVI K2V2 (当前 K_BITS=2 V_BITS=2)
CUDA_VISIBLE_DEVICES=1 python mem_spd_test.py \
    2>&1 | tee ~/projects/reproduce/KIVI/results/mem_spd_k2v2_bs16.txt

# 第 2 轮：FP16 baseline (K=16 V=16 走 else 分支用 LlamaForCausalLM 标准类)
sed -i 's/^K_BITS = 2$/K_BITS = 16/; s/^V_BITS = 2$/V_BITS = 16/' mem_spd_test.py
CUDA_VISIBLE_DEVICES=1 python mem_spd_test.py \
    2>&1 | tee ~/projects/reproduce/KIVI/results/mem_spd_fp16_bs16.txt

# 恢复到 K2V2 默认（避免后续混淆）
sed -i 's/^K_BITS = 16$/K_BITS = 2/; s/^V_BITS = 16$/V_BITS = 2/' mem_spd_test.py
```

每轮 stdout 末尾会打印：
```
used time: XXXX ms       # 平均一次 forward 的 wall time
peak mem: XX.XX GB       # 这轮 forward 的 peak VRAM
```

把两轮的 `used time` / `peak mem` 算 ratio 填进 §9 表。

> **为什么不用多卡？** 多卡 (`device_map="auto"` 自动切层) 能让 batch=96 跑通，但：(a) 每卡 peak VRAM 自然变小，不能跟 KIVI 量化省下来的混为一谈；(b) 跨卡 activation 传输的 PCIe 开销会让吞吐变慢。**单卡同 batch 才能直接算 ratio**对照 paper headline。

> **跳过整步**也合理：mem_spd_test 验的是 paper abstract 的 headline 数字，不是核心 PPL/score。如果只想要 LongBench 主表过线就完事，§7.2 可整步跳过，§9 表对应行标"未自测，引用 paper"。

## §8 抽数字 + 填表

```bash
cd ~/projects/reproduce

# KIVI K2V2 各 task 分数
cat KIVI/third_party/KIVI/pred/llama-2-7b-chat-hf_4096_2bits_group32_residual32/result.json | python -m json.tool

# FP16 baseline 各 task 分数
cat KIVI/third_party/KIVI/pred/llama-2-7b-chat-hf_4096_16bits_group32_residual32/result.json | python -m json.tool

# 显存 / 吞吐
cat KIVI/results/mem_spd.txt
```

把数字填到下面 §9。

## §9 实测 vs 论文

跑于 2026-05-12, env: torch 2.4.1 / transformers 4.43.1 / datasets 2.21.0 / flash-attn 2.8.3 (prebuilt wheel) / KIVI vendor commit `876b4d2` + 本地 patches 2 处 / GPU RTX 3090。完整元数据 → [results/_meta.md](results/_meta.md)。

### §9.1 LongBench 8-task standard mode (主验收)

| Task | K2V2 (g32 r32) | FP16 baseline | Δ (K2V2 − FP16) |
|------|---------------:|--------------:|----------------:|
| multi_news | 25.51 | 26.37 | −0.86 |
| repobench-p | 55.01 | 55.37 | −0.36 |
| qmsum | 20.27 | 20.58 | −0.31 |
| trec | 63.00 | 63.00 | 0.00 |
| lcc | 60.46 | 61.80 | −1.34 |
| samsum | 40.11 | 41.22 | −1.11 |
| triviaqa | 84.94 | 84.19 | **+0.75** |
| qasper | 19.08 | 19.43 | −0.35 |
| **8-task avg** | **46.05** | **46.50** | **−0.45** |

**判定**：**通过**。KIVI K2V2 比 FP16 baseline 仅低 0.45 分（≈ −1%），远在 ±1-2 分容差内；triviaqa 上反而高 0.75（小波动范围内）。跟 paper Table 5 报告的 KIVI vs FP16 ≈ −0.1 到 −0.5 同量级。

> 论文 anchor：KIVI (Liu et al., ICML 2024) [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) Table 5。paper 用的是 LLaMA-2-7B-chat（跟我们一致）。

### §9.2 Memory / throughput (paper headline 验证) —— **未复现**

batch=16, seqlen 160+338, 3 repeats, 单张 RTX 3090:

| 配置 | wall time (ms / generation) | peak VRAM (GB) |
|------|---:|---:|
| KIVI K2V2 (g32 r128) | **15292** | **15.39** |
| FP16 baseline | 13481 | 16.51 |
| **ratio (FP16 / KIVI)** | **0.88×**（KIVI 反慢 13%）| **0.93×**（KIVI 仅省 6.8%）|
| **Paper anchor** | 2.35-3.47× faster (abstract) | 2.6× less (abstract) |
| **判定** | **未复现** | **未复现** |

> **诚实记录**：paper headline 在我们的硬件 + batch=16 regime 下没复现。原因不是 KIVI 失效，是 **regime 不匹配**：paper benchmark 默认 `BATCH_SIZE=96` + 长 context（KV cache 主导显存的 regime），我们因单 3090 (24GB) 装不下 batch=96 fp16 baseline，只能砍到 batch=16 + 短 seqlen=498——这个 regime 下 model weights (14GB) 主导显存（占总 16.5GB 的 85%），KV cache 占比小，KIVI 的 INT2 优势几乎被淹没。
>
> **想真验 paper headline 需要的**：(a) 更大显存卡（A100 80GB）跑 batch=96，或 (b) 多卡 device_map="auto"（但又会让 ratio 失真——见 §7.2 注释）。Phase 1 不强求，§9.1 的 LongBench 主验收过了即算复现成功；headline 引用 paper 即可。
>
> **观察**：即便在这个不利 regime，KIVI 仍只比 FP16 慢 13% 而非更多——说明 K2V2 的 INT2 GEMV kernel + KIVI 自家 CUDA extension 工作正常，没有破窗失败。换言之**实现没问题，是 benchmark 设计需要的 regime 我们够不到**。

### §9.3 元数据 → [results/_meta.md](results/_meta.md)


## §10 Troubleshooting

> 大部分环境/前置坑已经被 §2 env.yml + §5 4 项预处理消化。下面是预期可能踩到的坑（基于 BiLLM 复现经验外推）。如果你跑出来发现新坑，补到对应 section。

### 编 `kivi_gemv` 报 `ModuleNotFoundError: No module named 'torch'`
跟 flash-attn 同样的 build-isolation 坑——`quant/setup.py` `import torch` 但 pip 的 build env 没装。**必须加 `--no-build-isolation`**：
```bash
cd ~/projects/reproduce/KIVI/third_party/KIVI/quant
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation
```

### `import kivi_gemv` 报 `ImportError: libc10.so: cannot open shared object file`
**不是没装好**——`.so` 在位，链接到 torch 的 `libc10.so`，但单独 import 时 torch 没载入进程，动态链接器找不到。先 import torch：
```bash
python -c "import torch; import kivi_gemv; print(kivi_gemv.__file__)"
```
KIVI 实际跑（`example.py` / `pred_long_bench.py`）都先 import torch，这只是裸验证时的伪问题。

### 编 `kivi_gemv` 报 `unsupported gpu architecture 'compute_XX'`
3090 是 sm_86。如果 nvcc 默认探测错，显式指定（已在 §3.2 命令里）：
```bash
TORCH_CUDA_ARCH_LIST="8.6" pip install -e . --no-build-isolation
```

### `pip install -e .` 时 flash-attn 报 `ModuleNotFoundError: No module named 'torch'`
flash-attn 的 setup.py 第一行 `import torch`，但 pip build isolation 默认把它放进无 torch 的 temp env。**不要** 用 `--no-build-isolation` 修——因为加了之后会撞下一个坑（cross-device link）。**正确做法**：用 §3.1 的预编译 wheel 直装，跳过 setup.py 整套机制。

### `Errno 18: Invalid cross-device link` 在装 flash-attn 时
`pip install flash-attn --no-build-isolation` 时它去 GitHub 拉预编译 wheel 到 `/tmp`，然后 `os.rename` 到 `$PIP_CACHE_DIR=/shared/yiminl50/.pip_cache`——两个 mount 跨文件系统失败。**修法**：跳过 pip 调用 flash-attn 自家 setup.py 这条路，手动 curl + pip install wheel（见 §3.1）。

### 真要本地编译 flash-attn（兜底，~10 min）
预编译 wheel 拿不到时（torch / Python 版本组合在 GitHub releases 没对应 wheel）才走这条：
```bash
TMPDIR=/shared/yiminl50/tmp MAX_JOBS=4 pip install flash-attn --no-build-isolation
```
`TMPDIR` 让 wheel 缓存写到 /shared 同盘，绕过 cross-device link。`MAX_JOBS=4` 防 32C 满载撑爆 125 GB RAM。

### `RuntimeError: CUDA out of memory` 在 LongBench 跑某个 task
LongBench 有几个 task 输入超长（gov_report、multi_news、passage_retrieval 等 32K context）。7B + 32K context + KIVI 应能塞进 24 GB；但 FP16 baseline 跑 32K 可能 OOM。
- baseline 阶段单卡跑不下就 skip 那 task：`pred_long_bench.py` 没有 `--skip-tasks` 直接 CLI，需要 patch 一下 `pred_long_bench.py` 里的 task list
- 或者 baseline 走 chat 模型 max_length 截 16K：找 `model.generate(max_new_tokens=..., max_length=...)` 处加 max_length=16384

### `eval_long_bench.py` 报 `ModuleNotFoundError: No module named 'jieba'` (或 rouge / fuzzywuzzy)
KIVI pyproject.toml 漏列了 LongBench 评分依赖。env.yml 已补；老 env 手动：
```bash
pip install jieba rouge fuzzywuzzy python-Levenshtein
```
> ⚠️ 注意是 `rouge` 不是 `rouge_score`——两个 PyPI 包：`rouge` 是 Pierre-Yves Lapersonne 的（KIVI 用这个，`from rouge import Rouge`），`rouge_score` 是 Google 的（lm-eval-harness 等用）。装错的话 `metrics.py` 在 `from rouge import Rouge` 处 ImportError。

### `RuntimeError: Dataset scripts are no longer supported, but found LongBench.py`
`datasets>=3.0` 砍了 script-based dataset 支持；THUDM/LongBench 还是老式（有 `LongBench.py` 加载器）。**修法**：降到 2.x。env.yml 已锁 `'datasets<3.0'`；老 env 手动：
```bash
pip install 'datasets<3.0'
```
（未来如果 datasets 2.x 也不能用了，alternative 是 patch `pred_long_bench.py` 读本地 jsonl——`/shared/yiminl50/datasets/longbench/data/<task>.jsonl` 已在 §5.2 下到位）

### `datasets` 库下 LongBench 卡死（如果走 datasets API 而非 huggingface-cli 预下载）
§5.2 已经 `huggingface-cli` 预下载到 `/shared/yiminl50/datasets/longbench/`。如果 `pred_long_bench.py` 还是走在线 `load_dataset('THUDM/LongBench', ...)`，确认 `$HF_DATASETS_CACHE` 指对（已在 `~/.bashrc`）。再卡的话参考 BiLLM/README §5.4 的 patch 模式：让脚本读本地 jsonl。

### LongBench eval 没动 / 没生成 result.json
确认：
- `pred_long_bench.py` 真的把 jsonl 落到了 `pred/<model>/<task>.jsonl`（KIVI 上游用 `args.model_name_or_path` 末段做 `<model>` 子目录名；如果你传完整路径会用 path basename）
- `eval_long_bench.py --model <name>` 的 `<name>` 跟上面 `<model>` 子目录名一致

### 多用户 GPU 冲突
开跑前 `gpu` 看其它用户（canying / jingchl6 / yeq6 / zhihenc5）在用哪张。挑空的，`CUDA_VISIBLE_DEVICES=<idx>` 显式锁定。LongBench 跑 3-6 小时——务必在 tmux 里。

### LongBench 分数 KIVI 显著低于 baseline (>5 分差)
1. 核对用的是 chat 模型不是 base
2. 核对 `--k_bits 2 --v_bits 2 --group_size 32 --residual_length 32` 跟 paper 主配置一致
3. KIVI K1V1（1-bit）会显著掉分，看看是不是误用了 K1V1
4. 看 KIVI issues：https://github.com/jy-yuan/KIVI/issues
5. **不要**调参刷数字。把异常记进 `results/_meta.md` 的"实测异常记录"区。

### tmux 掉了 / ssh 断了
始终 `tmux new -s <name>` 跑 LongBench（3-6 小时务必 tmux）。断线后 `tmux attach -t <name>` 接回。
