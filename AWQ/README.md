# AWQ — LLaMA-2-7B W4-g128 源码复现

> **核心理念**：从 `third_party/AutoAWQ` 源码 editable install (`pip install -e .`)，**不**用 `pip install autoawq` pip wheel。这样跑的就是 vendor 锁定那一版代码，可改可调。

> **上游状态**：AutoAWQ 已 [officially deprecated](https://github.com/casper-hansen/AutoAWQ#news-the-vllm-project-has-fully-adopted-autoawq)，最后官测 torch 2.6.0 / transformers 4.51.3。我们 vendor 一份固定版本，不受未来 API 变动影响。

## §0 目标 & 验收

| 项 | 值 |
|----|----|
| 模型 | `meta-llama/Llama-2-7b-hf` |
| 量化配置 | W4-g128, zero_point=True, version=GEMM |
| Calibration | upstream 默认（mit-han-lab/pile-val-backup, 128 samples） |
| 评测 | WikiText-2 PPL + 6 项 zero-shot（piqa, arc_e/c, hellaswag, winogrande, openbookqa） |
| 论文 anchor | WT2 PPL ≈ 5.60（AWQ paper Table 4），weights ≈ 3.7 GB |
| 验收 | WT2 PPL 与 anchor 差 ≤ ±0.3；weights 真 INT4 packed |

## §1 前置条件

- **GPU**：NVIDIA Compute Capability ≥7.5（Turing/Ampere/Ada/Hopper），≥16 GB VRAM 建议（量化峰值 ~14 GB）
- **CUDA driver**：≥12.1（PyTorch 2.4 + autoawq-kernels prebuilt 要求）
- **conda**：≥23.x
- **HuggingFace 账号**：已接受 Llama-2 许可（<https://huggingface.co/meta-llama/Llama-2-7b-hf>）

> **本地烟雾跑用 TinyLlama-1.1B**（不需 Llama-2 license），12 GB VRAM 即可。Canonical 跑放 lab 服务器。

## §2 创建 conda env

```bash
cd /path/to/quant/reproduce          # repo 根
bash scripts/env_lab.sh AWQ          # Linux lab
# 或 Windows: .\scripts\env_local.ps1 AWQ
conda activate quant-awq
```

`env.yml` 只装 PyTorch 2.4 + CUDA 12.1 + sentencepiece，**不**装 autoawq——故意留给源码 install。

## §3 从源码装 AutoAWQ（关键步骤）

```bash
cd AWQ/third_party/AutoAWQ
pip install -e ".[kernels,eval]"
```

含义：
- `-e`：editable install，源码改动立即生效（调试加 print 时必备）
- `[kernels]`：拉 `autoawq-kernels`（INT4 GEMM 推理 kernel）+ `flash-attn>=2.2`
- `[eval]`：拉 `lm_eval==0.4.1` + `tabulate` + `protobuf` + `evaluate` + `scipy`（跑上游 `examples/eval.py` 用）

`setup.py` 的 `install_requires` 会自动把 `transformers>=4.45 / triton / accelerate / datasets>=2.20 / huggingface_hub>=0.26.5 / zstandard / typing_extensions>=4.8 / tokenizers>=0.12.1` 一并装上。

> **flash-attn 编译慢警告**：`flash-attn>=2.2` 第一次装可能编译 5-15 分钟，吃满 CPU。如果只想先跑通，可改成 `pip install -e ".[eval]"` 跳过 kernels；后续再补。**纯量化不需要 kernels**（kernels 只影响推理速度，不影响 PPL 数字）。

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
git -C AWQ/third_party/AutoAWQ rev-parse HEAD
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

如果未设置 `$HF_HOME`：默认落 `~/.cache/huggingface/`。lab 上建议 `export HF_HOME=/shared/huggingface_cache` 共享缓存。

## §6 烟雾跑（本地，TinyLlama-1.1B，~5 min）

目的：验证整条 quantize → save → eval 链路通，**不**关心数字。

```bash
cd AWQ/third_party/AutoAWQ

mkdir -p ../../results/smoke

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
quant_path = '../../results/smoke/quantized'
quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Saved to {quant_path}')
"
```

**判定**：脚本不挂、`results/smoke/quantized/` 出现 `model.safetensors` + `config.json`。

> Smoke 产物默认不进 git（`.gitignore` 屏蔽 `*/results/smoke/`）。

## §7 Canonical 跑（lab，LLaMA-2-7B，~60 min）

### §7.1 量化（~30 min）

```bash
cd AWQ/third_party/AutoAWQ
mkdir -p ../../results

python -c "
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_path = 'meta-llama/Llama-2-7b-hf'
quant_path = '../../results/quantized_w4g128'
quant_config = {'zero_point': True, 'q_group_size': 128, 'w_bit': 4, 'version': 'GEMM'}

model = AutoAWQForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model.quantize(tokenizer, quant_config=quant_config)
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
print(f'Saved to {quant_path}')
" 2>&1 | tee ../../results/canonical_w4g128_quant_stdout.txt
```

> 量化中 GPU 峰值 ~14 GB；如果 OOM 看 §10。

### §7.2 PPL 评测（~5 min，用上游脚本）

```bash
# 在 AWQ/third_party/AutoAWQ 目录
python examples/eval.py \
    --model_path ../../results/quantized_w4g128 \
    --tasks wikitext \
    2>&1 | tee ../../results/canonical_w4g128_ppl_stdout.txt
```

> 上游 `examples/eval.py` 走 `awq.evaluation.evaluate_perplexity`（不是 lm-eval-harness 的 wikitext task），输出 WikiText-2 PPL 单值。

### §7.3 Zero-shot 评测（~25 min，用 lm-eval-harness CLI）

```bash
# 在 AWQ/third_party/AutoAWQ 目录
python -m lm_eval \
    --model hf \
    --model_args pretrained=../../results/quantized_w4g128,trust_remote_code=True \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande,openbookqa \
    --batch_size 1 \
    --output_path ../../results/canonical_w4g128_zeroshot.json \
    2>&1 | tee ../../results/canonical_w4g128_zeroshot_stdout.txt
```

> 这里 lm-eval 把 quantized model 当 HF causal LM 加载——AutoAWQ 已注册到 transformers AutoModel，所以 `--model hf` 直接能用。无需额外 hook。

### §7.4 Weight memory 测量

```bash
du -sh ../../results/quantized_w4g128/*.safetensors
# 期望: ~3.7 GB（INT4 packed），论文 anchor
```

## §8 抽数字 + 填表

```bash
# 从 zeroshot.json 抽 6 项 acc
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

| Config | Model | Metric | 实测 | 论文 anchor | 判定 |
|--------|-------|--------|------|------------|------|
| FP16 baseline | LLaMA-2-7B | WT2 PPL | _TBD_ | 5.47 | _baseline_ |
| W4-g128 | LLaMA-2-7B | WT2 PPL | _TBD_ | ≈ 5.60 | _TBD_ |
| W4-g128 | LLaMA-2-7B | piqa | _TBD_ | _TBD_ | _TBD_ |
| W4-g128 | LLaMA-2-7B | arc_easy | _TBD_ | _TBD_ | _TBD_ |
| W4-g128 | LLaMA-2-7B | arc_challenge | _TBD_ | _TBD_ | _TBD_ |
| W4-g128 | LLaMA-2-7B | hellaswag | _TBD_ | _TBD_ | _TBD_ |
| W4-g128 | LLaMA-2-7B | winogrande | _TBD_ | _TBD_ | _TBD_ |
| W4-g128 | LLaMA-2-7B | openbookqa | _TBD_ | _TBD_ | _TBD_ |
| W4-g128 | LLaMA-2-7B | weights GB | _TBD_ | ≈ 3.7 | _TBD_ |

> 论文 anchor 来源：AWQ paper (Lin et al., MLSys 2024) Table 4；FP16 baseline 来自同表第一行。

### §9.1 元数据（跑完填）

```markdown
- 模型 commit SHA: `<huggingface-cli scan-cache 看 meta-llama/Llama-2-7b-hf>`
- AutoAWQ vendor commit: `<git -C third_party/AutoAWQ rev-parse HEAD>`
- 环境: torch <版本>, transformers <版本>, autoawq 0.2.9
- GPU: <型号 + VRAM>
- 时间: <开始> → <结束>（量化 X min，eval Y min）
```

存到 `AWQ/results/canonical_w4g128_meta.md`。

## §10 Troubleshooting

### 装不上 flash-attn
跳过 kernels：`pip install -e ".[eval]"`。**不影响量化质量与 PPL**，只让推理慢。

### 量化时 OOM
- 确认没别的进程占 VRAM：`nvidia-smi`
- AutoAWQ 量化逐 transformer block 跑，`from_pretrained` 时加 `device_map='auto'` 让 accelerate 分流

### `import awq` 报 Triton 相关 ImportError
torch 2.4 + Triton 自带版本不兼容某些 GPU。临时 workaround：`pip install triton==2.3.1`。

### lm-eval 加载量化 model 报 `Unknown model type`
确认 transformers ≥ 4.45（autoawq setup.py 拉的版本应满足）；老版本 transformers 不识别 AWQ config。

### PPL 比 anchor 差 > 0.3
核对 model commit SHA、calibration 切片、上游 commit；再去 AutoAWQ issues 看已知 bug。**不要刷参数。**
