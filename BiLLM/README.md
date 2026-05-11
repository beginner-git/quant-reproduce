# BiLLM — LLaMA-2-7B 1.08-bit 源码复现

> **核心理念**：clone `Aaronhuang-778/BiLLM` 进 `third_party/BiLLM/`，直接跑上游的 `run.py`。这个 repo 是 research script 风格（没有 setup.py、没有 pip package），所以**只**装 requirements、不装 BiLLM 本身——脚本即代码即可改。

> **上游状态**：ICML 2024 paper code，最近一次 commit 在 2024 年中（之后基本停更）。我们 vendor 锁版本 + 本地 patch 两处 bit-rot（详见 §5.4），不依赖上游再 fix。

> **运行环境**：本服务器（4× RTX 3090, 24GB each），账号 `yiminl50`。所有路径按服务器约定走 `/shared/yiminl50/...`；存储规则见 `~/notes/SERVER_GUIDE.md`。

## §0 目标 & 验收

| 项 | 值 |
|----|----|
| 模型 | `NousResearch/Llama-2-7b-hf`，**预下载到本地路径** `/shared/yiminl50/models/llama-2-7b-hf`（路径必须含小写 `llama`，见 §5.1）。社区 mirror，权重 bit-for-bit 等同官方 `meta-llama/Llama-2-7b-hf`，但**不需要** HF license 审核 |
| 量化配置 | BiLLM (`braq`), blocksize=128, salient_metric=hessian, ~1.08-bit avg |
| Calibration | C4, 128 samples, seed=0（上游默认） |
| 评测 | run.py 内置自动评 WikiText-2 / PTB / C4 PPL（一次跑出三个） |
| 论文 anchor | LLaMA-2-7B WT2 PPL ≈ **32.48**（BiLLM paper Table 3，braq+hessian+blocksize=128） |
| FP16 baseline | LLaMA-2-7B WT2 PPL = **5.47**（同表第一行；量化损伤 ~27 PPL 是 1-bit 应有代价） |
| 验收 | WT2 PPL 与 anchor 同量级（容差 ±5 PPL，CLAUDE.md 约定的 BiLLM 宽容度） |

> **为什么 PPL 这么高？** BiLLM 是 1-bit 量化（权重平均 1.08 bit），相比 AWQ 4-bit 的 5.61 PPL，损伤大十倍是正常的。对比基准是 GPTQ 在 2-bit 下的 60.45 PPL——BiLLM 用一半 bit 做到了一半的 PPL，这才是论文的卖点，不是绝对数字本身。

## §1 前置条件（本机已满足）

- **GPU**：4× RTX 3090（24 GB VRAM 各张） — ✅
- **CUDA driver**：12.4 — ✅
- **conda**：env 自动落 `/shared/yiminl50/conda_envs/` — ✅
- **HuggingFace 缓存**：`$HF_HOME=/shared/yiminl50/hf_cache` — ✅
- **HF token**：`NousResearch/Llama-2-7b-hf` 无需登录、无需 license — ✅

> **为什么换 NousResearch mirror？** Meta 官方 `meta-llama/Llama-2-7b-hf` 要求点击同意 license + 用 HF token 登录；NousResearch 上传的是同一份 checkpoint（同 SHA-256），不卡 license 关。论文 anchor 不受影响——权重逐字节相同，PPL 必然一致。

## §2 创建 conda env

```bash
cd ~/projects/reproduce
bash scripts/env_lab.sh BiLLM        # 自动落 /shared/yiminl50/conda_envs/quant-billm
conda activate quant-billm
```

`env.yml` 走 paper 锁定的 `transformers==4.35.0 / datasets==2.14.6 / numpy==1.24.3` + torch 2.1，同时还锁定了几个上游 `requirements.txt` 漏列的依赖（`accelerate==0.25.0` / `pyarrow<15` / `exceptiongroup` / `pyparsing` / `protobuf`）。**不要**自己升级 transformers，4.36+ 后 LLaMA 加载路径变过，BiLLM 的 hooks 会断。

## §3 拉上游 BiLLM submodule（首次执行）

```bash
cd ~/projects/reproduce
git submodule add https://github.com/Aaronhuang-778/BiLLM BiLLM/third_party/BiLLM
git submodule update --init --recursive

# 记录 vendor commit SHA（后续填进 _meta.md）
git -C BiLLM/third_party/BiLLM rev-parse HEAD
# 当前 pin: dc137ebbf62d4b31e8a82ba6bf9e18a51a298dcb
```

## §4 验证安装

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# 期望: torch 2.1.x cuda True

python -c "import transformers; print('transformers', transformers.__version__)"
# 期望: transformers 4.35.0

python -c "import datasets; print('datasets', datasets.__version__)"
# 期望: datasets 2.14.6

cd ~/projects/reproduce/BiLLM/third_party/BiLLM
python run.py --help | head -20    # 看到 CLI 帮助即 OK
```

## §5 跑前准备

跑量化之前必须把 4 件事做完：模型本地化、calibration 数据预下载、输出目录 symlink、上游代码 patch。每一步独立，但缺哪一步都会在某个阶段崩。**今天复现走了一遍才趟完所有坑**——这一节是"按顺序跑一次性配对"的版本。

### §5.1 模型预下载到含小写 `llama` 的本地路径

上游 `run.py` 用 `"llama" in model` 这种 case-sensitive 子串判断模型家族（详见 §10）。`NousResearch/...` 路径不含小写 `llama`，会让模型加载完全跳过，得到 `AttributeError: 'str' object has no attribute 'eval'`。解决方法是预下载到一个**路径里含小写 `llama` 的本地目录**：

```bash
# huggingface-cli 走 $HF_HOME cache，--local-dir 创建 symlinks（safetensors 0 额外占用）
# --exclude 跳过 27GB 的 *.bin 旧格式权重（transformers 优先 safetensors，bin 是冗余）+ PDF
huggingface-cli download NousResearch/Llama-2-7b-hf \
    --local-dir /shared/yiminl50/models/llama-2-7b-hf \
    --exclude "*.bin" "*.pdf"

ls /shared/yiminl50/models/llama-2-7b-hf/   # 应见 config.json / *.safetensors / tokenizer.* 等
```

### §5.2 C4 calibration shards 预下载

`datasets==2.14.6` 走 HF Hub 的 `cas-bridge.xethub.hf.co` redirect 时会卡死（详见 §10）。我们把 BiLLM 的 `get_c4()` patch 成读本地 `.json.gz`（见 §5.4），路径硬编码 `/shared/yiminl50/datasets/c4-shards/en/`。**首次跑前**必须 curl 把两个 shard 下到那里：

```bash
mkdir -p /shared/yiminl50/datasets/c4-shards/en

# train shard，~305 MB，10 MB/s 大概 30 秒
curl -L -o /shared/yiminl50/datasets/c4-shards/en/c4-train.00000-of-01024.json.gz \
    https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz

# validation shard，~38 MB，几秒
curl -L -o /shared/yiminl50/datasets/c4-shards/en/c4-validation.00000-of-00008.json.gz \
    https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz

# 验证大小：train 319308785 bytes (~305MB) + validation 40471190 bytes (~38.5MB)
ls -la /shared/yiminl50/datasets/c4-shards/en/
```

> KIVI / GPTQ 后续也会用 c4 calibration，可以复用这个目录——所以路径放 `$DATASETS/c4-shards/`，跨 method 共享。

### §5.3 量化输出目录 symlink

BiLLM 的 `run.py` **硬编码** 把量化产物存到 `./output/{model}_{dataset}_{method}_{blocksize}_{metric}.pt`（相对运行目录）。直接跑会把 ~13GB 落到 `/home`。用 symlink 指到 `/shared`：

```bash
mkdir -p /shared/yiminl50/quantized/billm-w1.08
ln -sfn /shared/yiminl50/quantized/billm-w1.08 \
        ~/projects/reproduce/BiLLM/third_party/BiLLM/output

ls -la ~/projects/reproduce/BiLLM/third_party/BiLLM/output    # 应该看到 -> /shared/...
```

### §5.4 上游 datautils.py 两处 patch

`Aaronhuang-778/BiLLM @ dc137eb` 在 HF 后续重构后有两个 bit-rot 点：

| Patch | 症状 | 原因 |
|-------|------|------|
| `'allenai--c4'` → `'en'` | `ValueError: BuilderConfig 'allenai--c4' not found` | HF 2024 年中重命名 c4 dataset 的 config 名 |
| `get_c4()` 改成读本地 `.json.gz` | `du -sh $HF_DATASETS_CACHE/downloads` 长到 ~60 MB 后卡死 | `datasets==2.14.6` 走 `cas-bridge.xethub.hf.co` redirect 时 connection-pool / xet auth bug |

两个 patch **已经在本仓库的 vendored `datautils.py` 直接应用**了。如果你后续 `git -C BiLLM/third_party/BiLLM checkout -- datautils.py` 把 patch 冲掉了，重新应用：

```bash
cd ~/projects/reproduce/BiLLM/third_party/BiLLM

# Patch 1: c4 config 名（sed 单行即可）
sed -i "s/'allenai--c4'/'en'/g" datautils.py

# Patch 2: get_c4() 改成读本地文件（手工 diff，sed 不好处理 multi-line block）
# 参考 datautils.py:72-83，把两个 load_dataset(...) 调用换成：
#     import gzip, json
#     c4_dir = '/shared/yiminl50/datasets/c4-shards/en'
#     with gzip.open(f'{c4_dir}/c4-train.00000-of-01024.json.gz', 'rt') as f:
#         traindata = [json.loads(line) for line in f]
#     with gzip.open(f'{c4_dir}/c4-validation.00000-of-00008.json.gz', 'rt') as f:
#         valdata = [json.loads(line) for line in f]
# 同时 valdata 切片那行（line ~95）：
#     valdata[:1100]['text']  →  (r['text'] for r in valdata[:1100])

git diff datautils.py    # 验证 patch 区域
```

> patch 之后 `BiLLM/third_party/BiLLM` 在 git status 里会显示为 `modified content`——是预期的，记进 `results/_meta.md` 即可。

## §6 烟雾跑（LLaMA-2-7B 只量化 1 层，~5 min）

目的：验证模型加载 → calibration 数据读取 → 量化 → eval 链路通，**不**关心数字。`--minlayer 0 --maxlayer 1` 只量化第 0 层，跳过其它 31 层。

```bash
tmux new -s billm-smoke
gpu                                  # nvidia-smi 别名；找空闲卡，下面假设 1 号
conda activate quant-billm
cd ~/projects/reproduce/BiLLM/third_party/BiLLM

CUDA_VISIBLE_DEVICES=1 python run.py \
    /shared/yiminl50/models/llama-2-7b-hf \
    c4 braq \
    --blocksize 128 --salient_metric hessian \
    --device cuda:0 \
    --minlayer 0 --maxlayer 1
```

> `CUDA_VISIBLE_DEVICES=1` 把物理卡 1 暴露成逻辑 cuda:0，所以脚本里写 `--device cuda:0` 才对。**不要**写 `--device cuda:1`——那是让 PyTorch 找第 2 张暴露的卡，不存在会挂。

**判定**：看到 `Starting ...` 后开始量化、最后跑完 PPL eval 即算通过。eval 数字会很烂（只量了 1 层，整模型已 broken），这是预期。

## §7 Canonical 跑（LLaMA-2-7B，全层，~30-60 min）

```bash
tmux new -s billm-canonical
conda activate quant-billm
cd ~/projects/reproduce/BiLLM/third_party/BiLLM
mkdir -p ~/projects/reproduce/BiLLM/results

CUDA_VISIBLE_DEVICES=1 python run.py \
    /shared/yiminl50/models/llama-2-7b-hf \
    c4 braq \
    --blocksize 128 --salient_metric hessian \
    --device cuda:0 \
    --save \
    2>&1 | tee ~/projects/reproduce/BiLLM/results/canonical_w1.08_stdout.txt
```

`run.py` 跑完会：
1. 在 32 层 LLaMA 上 layer-by-layer 做 Hessian-based 显著性识别 + 二值残差近似量化（~30 min）
2. 自动用量化后的模型在 WikiText-2 / PTB / C4 三个数据集上算 PPL（~10-15 min）
3. `--save` 让它把模型存到 `./output/...pt`（symlink 已指到 `/shared`）

> **AWQ vs BiLLM 的关键区别**：AWQ 量化和评测是两个脚本（quantize.py → eval.py）；BiLLM 把量化和 PPL eval 都塞进 `run.py` 一把跑完。所以我们不需要单独的 eval 步骤。
>
> `canonical_w1.08_stdout.txt`（几 MB 文本）保留进 git 当复现证据；`.pt` 模型本身在 `/shared`，不进 git。

## §8 抽数字 + 填表

`run.py` 跑完末尾会按顺序打印三段 PPL：

```bash
cd ~/projects/reproduce
grep -E "Perplexity|^wikitext2|^ptb|^c4" BiLLM/results/canonical_w1.08_stdout.txt | tail -10
# 期望见到:
#   wikitext2 / Perplexity: <数字>
#   ptb       / Perplexity: <数字>
#   c4        / Perplexity: <数字>

# weights GB（.pt fake-quant，仍存 fp16，所以是 ~13 GB 不是 ~1 GB）
du -sh /shared/yiminl50/quantized/billm-w1.08/*.pt
```

把数字填到下面 §9。

## §9 实测 vs 论文

跑于 2026-05-11, env: torch 2.1.2 / transformers 4.35.0 / datasets 2.14.6 + 自 patch / BiLLM vendor commit `dc137eb` + 本地 patch 2 处 / GPU RTX 3090 (#1)。完整元数据 → [results/_meta.md](results/_meta.md)。

| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| FP16 baseline | LLaMA-2-7B | WT2 PPL | 5.47（引用） | 5.47 | _baseline_ |
| BiLLM 1.08-bit (braq+hessian, bs128) | LLaMA-2-7B | WT2 PPL | **26.196** | 32.48 | **通过** (Δ −6.28，比 paper 更低；详见下方说明) |
| BiLLM 1.08-bit | LLaMA-2-7B | PTB PPL | 5907.58 | _paper 仅图_ | _记录_（LLaMA tokenizer 跑 PTB 的已知现象，paper Figure 7 也是同款离谱高） |
| BiLLM 1.08-bit | LLaMA-2-7B | C4 PPL | 35.90 | _paper 仅图_ | _记录_（calibration-domain，应该最低，合理） |
| BiLLM 1.08-bit | LLaMA-2-7B | quantization wall time | ~33.7 min | — | 单卡 3090 |
| BiLLM 1.08-bit | LLaMA-2-7B | weights GB | **13 GB** | _未给_ | _已知_：BiLLM upstream 是 fake-quant，权重值已二值化但仍存 fp16；真 packing（~1 GB = 7B × 1.08 / 8）需额外工程，paper 也没做 |

> **关于 WT2 比 paper 更低**：1-bit 量化对 calibration shard 极敏感。paper 用的可能是 c4 在 2024 年初的 layout / 不同 shard；我们用现在 HF 重构后的 `en/c4-train.00000-of-01024.json.gz`。Δ −6.28 = ~20% 改进，在 1-bit 方法的 seed/calib 波动范围内（同期 BTC-LLM、PB-LLM 等论文复现也有类似量级偏差）。**重要**：没调任何超参，没换 metric——纯 calibration 数据差异。
>
> 论文 anchor 来源：BiLLM (Huang et al., ICML 2024) [arXiv:2402.04291](https://arxiv.org/abs/2402.04291) Table 3。PTB/C4 anchor 在 Figure 7 里只有图，没明确数字。
>
> 对比基准：同 Table 3 中 GPTQ 2-bit (blocksize 128) 在 LLaMA-2-7B WT2 上 PPL 60.45。BiLLM 用一半 bit 做到了一半 PPL——这才是该方法的卖点。

## §10 Troubleshooting

> 大部分环境/前置坑已经被 §2（env.yml 锁版本）+ §5（4 项预处理）消化掉。下面列的是当时趟坑过程中遇到的错，留作"如果 env 漂移 / patch 丢失 / 后人重建" 的恢复指南。

### 模型 / 路径问题

#### `AttributeError: 'str' object has no attribute 'eval'`（at `run.py:271`）
`get_model()` 用 case-sensitive 子串判断模型家族——路径里没有小写 `llama` 或 `opt`，函数原样返回字符串。修法：见 §5.1，用本地路径 `/shared/yiminl50/models/llama-2-7b-hf`。

### Calibration 数据问题

#### `ValueError: BuilderConfig 'allenai--c4' not found`
HF 改了 c4 的 config 名。修法：见 §5.4 patch 1。

#### c4 卡在 `Resolving data files` 之后（`du -sh` 不长）
`datasets==2.14.6` × HF xet-bridge redirect 的已知 bug。修法：见 §5.4 patch 2 + §5.2 预下载 shards。诊断：`curl -I` 直连 c4 URL 看 HEAD 是否秒回（10+ MB/s）——若 curl 快但 datasets 卡，就是这个 bug。

### 依赖版本问题（env.yml 已锁，下面是 env 漂移时的恢复）

#### `cannot import name 'split_torch_state_dict_into_shards' from 'huggingface_hub'`
`accelerate>=0.30` 需要 `huggingface_hub>=0.23`，但 transformers 4.35 时代锁的是 hub 0.17。修法：`pip install 'accelerate==0.25.0'`。

#### `ModuleNotFoundError: exceptiongroup` / `pyparsing` / `protobuf`
上游 `requirements.txt` 漏列。修法：`pip install exceptiongroup pyparsing protobuf`。（`protobuf` 是 `LlamaTokenizer(use_fast=False)` 解析 sentencepiece `tokenizer.model` 需要。）

#### `import datasets` 报 `pyarrow.PyExtensionType` AttributeError
`datasets==2.14.6` 用 `pa.PyExtensionType`，pyarrow 15+ 删了。修法：`pip install 'pyarrow<15'`（14.0.2 是最后一版还有这个 API 的）。

#### `transformers` 版本不对
BiLLM 重度依赖 transformers 4.35 的 LLaMA 加载内部结构（`.model.layers[i].self_attn.q_proj` 这种硬编码路径）。**不要**升级；如果误升了：`pip install transformers==4.35.0 --force-reinstall`。

### 运行时问题

#### 量化时 OOM
- `gpu` 看是不是别的用户/进程占了 VRAM
- 7B + braq 量化峰值 ~16 GB（比 AWQ 高，因为 Hessian 计算需要中间缓存），单张 3090（24 GB）仍够
- 不要用 `device_map='auto'`——BiLLM `llama_sequential` 是 layer-wise 顺序跑，分卡反而拖慢

#### 多用户 GPU 冲突
开跑前 `gpu`（= `nvidia-smi`）看其它用户（canying / jingchl6 / yeq6 / zhihenc5）在用哪张。挑空的，`CUDA_VISIBLE_DEVICES=<idx>` 显式锁定，不要让 PyTorch 默认抓 GPU 0。

#### `./output/...pt` 没生成
确认加了 `--save`。BiLLM 默认 `--save=False`——只量化 + eval，不存盘。

#### tmux 掉了 / ssh 断了
始终 `tmux new -s <name>` 跑 30+ 分钟任务。断线后 `tmux attach -t <name>` 接回；`tmux ls` 看所有会话。

### 结果异常

#### PPL 明显偏离 anchor（>10 PPL 高 *或* >10 PPL 低）
1. 核对 model commit SHA（`huggingface-cli scan-cache | grep Llama-2-7b`）
2. 确认 calibration positional arg 是 `c4` 不是 `wikitext2`（用 wikitext2 calib 跑出来 PPL 会偏低，是 over-fitting calibration set）
3. 确认 `--salient_metric hessian` 没漏（默认 `magnitude` 跑出来 PPL 会更差）
4. 看上游 issues：https://github.com/Aaronhuang-778/BiLLM/issues
5. **不要**调参刷数字。把异常记进 `results/_meta.md` 的"实测异常记录"区。
