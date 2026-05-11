# LLM 量化论文复现

复现近年 LLM 量化方向的代表论文，从源码出发，附评测数字与研读笔记。

## 方法

| 方法 | 论文 | 状态 | 链接 |
|------|------|------|------|
| **AWQ**   | Lin et al., MLSys 2024     | 进行中 | [AWQ/](AWQ/) |
| **BiLLM** | Huang et al., ICML 2024    | 待开始 | — |
| **KIVI**  | Liu et al., ICML 2024      | 待开始 | — |
| **GPTQ**  | Frantar et al., ICLR 2023  | 待开始 | — |

## 数字汇总

见 [`docs/results/summary.md`](docs/results/summary.md)。

## 环境

- **运行机**：GPU 服务器（4× RTX 3090, 24GB each），账号 `yiminl50`；存储约定见 `~/notes/SERVER_GUIDE.md`。
- **conda env 隔离**：每方法独立 env（`quant-awq` / `quant-billm` / `quant-kivi` / `quant-gptq`），avoid 依赖冲突。env 通过 `~/.condarc` 自动落 `/shared/yiminl50/conda_envs/`。
- **HF cache**：`$HF_HOME=/shared/yiminl50/hf_cache`（已在 `~/.bashrc` 配置），模型权重跨 env 共享。
- **入口**：操作手册 [`docs/howto-reproduce.md`](docs/howto-reproduce.md)；建 env 用 `scripts/env_lab.sh <METHOD>`。
