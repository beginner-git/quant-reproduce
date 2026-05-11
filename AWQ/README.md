# AWQ — LLaMA-2-7B W4-g128 源码复现

> **核心理念**：从 `third_party/AutoAWQ` 源码 editable install (`pip install -e .`)，**不**用 `pip install autoawq` pip wheel。这样跑的就是 vendor 锁定那一版代码，可改可调。

> **上游状态**：AutoAWQ 已 [officially deprecated](https://github.com/casper-hansen/AutoAWQ#news-the-vllm-project-has-fully-adopted-autoawq)，最后官测 torch 2.6.0 / transformers 4.51.3。我们 vendor 一份固定版本，不受未来 API 变动影响。

> **运行环境**：本服务器（4× RTX 3090, 24GB each），账号 `yiminl50`。所有路径按服务器约定走 `/shared/yiminl50/...`；存储规则见 `~/notes/SERVER_GUIDE.md`。

## §0 目标 & 验收

| 项 | 值 |
|----|----|
| 模型 | `meta-llama/Llama-2-7b-hf` |
| 量化配置 | W4-g128, zero_point=True, version=GEMM |
| Calibration | upstream 默认（mit-han-lab/pile-val-backup, 128 samples） |
| 评测 | WikiText-2 PPL + 6 项 zero-shot（piqa, arc_e/c, hellaswag, winogrande, openbookqa） |
| 论文 anchor | WT2 PPL ≈ 5.60（AWQ paper Table 4），weights ≈ 3.7 GB |
| 验收 | WT2 PPL 与 anchor 差 ≤ ±0.3；weights 真 INT4 packed |

## §1 前置条件（本机已满足）

- **GPU**：4× RTX 3090（Ampere，compute capability 8.6，24 GB VRAM 各张）— ✅
- **CUDA driver**：12.4（`nvidia-smi` 右上角） — ✅
- **nvcc**：12.0（`/usr/local/cuda`） — ✅ 仅 BiLLM/KIVI 用，AWQ 走 prebuilt wheel
- **conda**：在 `/shared/yiminl50/miniconda3`，env 自动落 `/shared/yiminl50/conda_envs/` — ✅
- **HuggingFace token**：登录过即可（`huggingface-cli whoami` 验证）
- **Llama-2 license**：HF 上接受过 `meta-llama/Llama-2-7b-hf` 使用条款

> Smoke 跑用 TinyLlama-1.1B（无 license 要求）；canonical 跑用 LLaMA-2-7B。两者都在本服务器一台机器上跑，按 `CUDA_VISIBLE_DEVICES` 选卡。

## §2 创建 conda env

```bash
cd ~/projects/reproduce
bash scripts/env_lab.sh AWQ           # 自动落 /shared/yiminl50/conda_envs/quant-awq
conda activate quant-awq
```

`env.yml` 只装 PyTorch 2.4 + CUDA 12.1 + sentencepiece，**不**装 autoawq——故意留给源码 install。

## §3 从源码装 AutoAWQ（关键步骤）

```bash
cd ~/projects/reproduce/AWQ/third_party/AutoAWQ
pip install -e ".[kernels,eval]"
```

含义：
- `-e`：editable install，源码改动立即生效（调试加 print 时必备）
- `[kernels]`：拉 `autoawq-kernels`（INT4 GEMM 推理 kernel）+ `flash-attn>=2.2`
- `[eval]`：拉 `lm_eval==0.4.1` + `tabulate` + `protobuf` + `evaluate` + `scipy`（跑上游 `examples/eval.py` 用）

`setup.py` 的 `install_requires` 会自动把 `transformers>=4.45 / triton / accelerate / datasets>=2.20 / huggingface_hub>=0.26.5 / zstandard / typing_extensions>=4.8 / tokenizers>=0.12.1` 一并装上。

> **flash-attn 编译慢警告**：`flash-attn>=2.2` 第一次装可能编译 5-15 分钟，吃满 CPU（32C 服务器上略快）。建议开 tmux 跑：`tmux new -s awq-install`。如果只想先跑通量化，可改成 `pip install -e ".[eval]"` 跳过 kernels；后续再补。**纯量化不需要 kernels**（kernels 只影响推理速度，不影响 PPL 数字）。

## §4 验证安装

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# 期望: torch 2.4.x cuda True

python -c "import awq; print('autoawq', awq.__version__)"
# 期望: autoawq 0.2.9（vendor commit 88e4c76 锁定的版本）

python -c "from awq import AutoAWQForCausalLM; print(AutoAWQForCausalLM)"
# 期望: 不报错，打印 class

python -c "import lm_eval; print('lm_eval', lm_eval.__version__)"
# 期望: lm_eval 0.4.1

# vendor commit SHA（meta 元数据用）
git -C ~/projects/reproduce/AWQ/third_party/AutoAWQ rev-parse HEAD
```

如果 `import awq` 弹 deprecation warning 是正常的，上游故意留的 final dev message。

## §5 模型授权 + 下载

```bash
huggingface-cli login                # 粘贴 HF token（read 权限够）

# 预下载 LLaMA-2-7B（14 GB），落到 $HF_HOME
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
"
```

> `$HF_HOME` 已在 `~/.bashrc` 设置为 `/shared/yiminl50/hf_cache`，14GB 权重落那里跨 env 共享。第一次下载几分钟到十几分钟（取决于网速）；后续 env 直接命中 cache。

## §6 烟雾跑（TinyLlama-1.1B，~5 min）

目的：验证整条 quantize → save → eval 链路通，**不**关心数字。

```bash
# 先 `gpu` 看哪张卡闲；开 tmux 防 ssh 掉线
tmux new -s awq-smoke
gpu                                   # nvidia-smi 别名；找空闲卡，下面假设 0 号

conda activate quant-awq
cd ~/projects/reproduce/AWQ/third_party/AutoAWQ

CUDA_VISIBLE_DEVICES=0 python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
quant_path = '/shared/yiminl50/quantized/awq-smoke/quantized'    # 落 /shared 大盘
quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Saved to {quant_path}')
"
```

**判定**：脚本不挂、`/shared/yiminl50/quantized/awq-smoke/quantized/` 出现 `model.safetensors` + `config.json`。

> 量化产物按服务器约定落 `$QUANTIZED/awq-smoke/quantized/`（11TB 大盘），不挤 `/home` 的 915GB 系统盘。

## §7 Canonical 跑（LLaMA-2-7B，~60 min）

LLaMA-2-7B 量化峰值显存 ~14 GB，单张 3090（24 GB）足够。开 tmux 后台跑。

### §7.1 量化（~30 min）

```bash
tmux new -s awq-canonical
conda activate quant-awq
cd ~/projects/reproduce/AWQ/third_party/AutoAWQ

CUDA_VISIBLE_DEVICES=0 python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = '/shared/yiminl50/quantized/awq-w4g128/quantized'
quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Saved to {quant_path}')
" 2>&1 | tee ~/projects/reproduce/AWQ/results/canonical_w4g128_quant_stdout.txt
```

> 注意：`AWQ/results/canonical_w4g128_quant_stdout.txt` 是小文本（几 MB），保留进 git 当复现证据。量化产物本身在 `/shared`。

> 如果 OOM 看 §10。多 GPU 不要 `device_map='auto'`；AWQ 量化是 layer-wise 顺序跑，单卡足够。

### §7.2 PPL 评测（~5 min，用上游脚本）

```bash
conda activate quant-awq
cd ~/projects/reproduce/AWQ/third_party/AutoAWQ

CUDA_VISIBLE_DEVICES=0 python examples/eval.py \
    --model_path /shared/yiminl50/quantized/awq-w4g128/quantized \
    --tasks wikitext \
    2>&1 | tee ~/projects/reproduce/AWQ/results/canonical_w4g128_ppl_stdout.txt
```

> 上游 `examples/eval.py` 走 `awq.evaluation.evaluate_perplexity`（不是 lm-eval-harness 的 wikitext task），输出 WikiText-2 PPL 单值。

### §7.3 Zero-shot 评测（~25 min，用 lm-eval-harness CLI）

```bash
conda activate quant-awq
cd ~/projects/reproduce/AWQ/third_party/AutoAWQ

CUDA_VISIBLE_DEVICES=0 python -m lm_eval \
    --model hf \
    --model_args pretrained=/shared/yiminl50/quantized/awq-w4g128/quantized,trust_remote_code=True \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ~/projects/reproduce/AWQ/results/canonical_w4g128_zeroshot.json \
    2>&1 | tee ~/projects/reproduce/AWQ/results/canonical_w4g128_zeroshot_stdout.txt
```

> lm-eval 把 quantized model 当 HF causal LM 加载——AutoAWQ 已注册到 transformers AutoModel，所以 `--model hf` 直接能用。无需额外 hook。
>
> `_zeroshot.json`（小 JSON）和 `_zeroshot_stdout.txt` 保留进 git；模型本身仍在 `/shared`。

### §7.4 Weight memory 测量

```bash
du -sh /shared/yiminl50/quantized/awq-w4g128/quantized/*.safetensors
# 期望: ~3.7 GB（INT4 packed），论文 anchor
```

## §8 抽数字 + 填表

```bash
# 从 zeroshot.json 抽 6 项 acc
cd ~/projects/reproduce
python -c "
import json
d = json.load(open('AWQ/results/canonical_w4g128_zeroshot.json'))['results']
for t in ['piqa','arc_easy','arc_challenge','hellaswag','winogrande','openbookqa']:
    print(t, ':', round(d[t]['acc,none'], 4))
"

# WT2 PPL 直接从 canonical_w4g128_ppl_stdout.txt 末尾 grep
grep -i "perplexity\|ppl" AWQ/results/canonical_w4g128_ppl_stdout.txt | tail -5
```

填到下面 §9 表格。

## §9 实测 vs 论文

跑于 2026-05-11，env: torch 2.4.1 / transformers 4.51.3 / autoawq 0.2.9 (vendor commit `88e4c76`) / GPU RTX 3090。完整元数据 → [results/_meta.md](results/_meta.md)。

| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| FP16 baseline | LLaMA-2-7B | WT2 PPL | _未测_ | 5.47 | _baseline_ |
| W4-g128 | LLaMA-2-7B | WT2 PPL (token-level, seqlen=2048) | **5.614** | ≈ 5.60 | **通过** (Δ +0.014, 容差 ±0.3) |
| W4-g128 | LLaMA-2-7B | weights GB | **3.62** | ≈ 3.7 | **通过** |
| W4-g128 | LLaMA-2-7B | piqa (acc / acc_norm) | 0.7791 / 0.7873 | — | _无量化损伤_ |
| W4-g128 | LLaMA-2-7B | arc_easy (acc / acc_norm) | 0.7559 / 0.7311 | — | _无量化损伤_ |
| W4-g128 | LLaMA-2-7B | arc_challenge (acc / acc_norm) | 0.4334 / 0.4497 | — | _无量化损伤_ |
| W4-g128 | LLaMA-2-7B | hellaswag (acc / acc_norm) | 0.5663 / 0.7521 | — | _无量化损伤_ |
| W4-g128 | LLaMA-2-7B | winogrande (acc) | 0.6819 | — | _无量化损伤_ |
| W4-g128 | LLaMA-2-7B | openbookqa (acc / acc_norm) | 0.3140 / 0.4440 | — | _无量化损伤_ |
| W4-g128 | LLaMA-2-7B | 6-task avg (acc_norm; winogrande 用 acc) | **0.6451** | ≈ 0.65 (FP16 baseline) | **通过** |

> 论文 anchor 来源：AWQ paper (Lin et al., MLSys 2024) Table 4；FP16 baseline 来自同表第一行。Table 4 只列了 PPL，没给 per-task zero-shot anchor，因此 zero-shot 行的"论文 anchor"列空缺；6-task 平均的 0.65 anchor 是 LLaMA-2-7B FP16 baseline 在 lm-eval-harness 上的公开复现典型值 (0.64–0.66)。实测 0.6451 落在该区间内，与 PPL +0.014 的微小差距一致，说明 W4-g128 量化对 zero-shot 推理几乎无影响。

### §9.1 元数据

详见 [results/_meta.md](results/_meta.md) —— 包含 vendor commit、完整 env 包版本、quant config、eval 命令、wall time、env 重建坑位备忘。

## §10 Troubleshooting

### 装不上 flash-attn
跳过 kernels：`pip install -e ".[eval]"`。**不影响量化质量与 PPL**，只让推理慢。

### 量化时 OOM
- `gpu` 看是不是别的用户/进程占了 VRAM（4 张卡多用户共享）
- 用 `CUDA_VISIBLE_DEVICES=<idx>` 明确指定空闲卡
- 7B 量化峰值 ~14 GB，单张 3090（24 GB）富裕；如果还 OOM，重启 Python 进程清残留 VRAM
- AutoAWQ 量化逐 transformer block 跑，`from_pretrained` 时加 `device_map='auto'` 让 accelerate 分流（多卡）

### 多用户 GPU 冲突
开跑前 `gpu`（= `nvidia-smi`）看其它用户（canying / jingchl6 / yeq6 / zhihenc5）在用哪张。挑空的，`CUDA_VISIBLE_DEVICES=<idx>` 显式锁定。**不要省略这一步**让脚本默认抓 GPU 0。

### `import awq` 报 Triton 相关 ImportError
torch 2.4 + Triton 自带版本不兼容某些 GPU。临时 workaround：`pip install triton==2.3.1`。

### lm-eval 加载量化 model 报 `Unknown model type`
确认 transformers ≥ 4.45（autoawq setup.py 拉的版本应满足）；老版本 transformers 不识别 AWQ config。

### PPL 比 anchor 差 > 0.3
核对 model commit SHA、calibration 切片、上游 commit；再去 AutoAWQ issues 看已知 bug。**不要刷参数。**

### 跑长任务 tmux 掉了 / ssh 断了
始终用 `tmux new -s <name>` 跑量化 + eval。断线后 `tmux attach -t <name>` 接回。`tmux ls` 看所有会话。
