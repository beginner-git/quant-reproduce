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

每方法独立 conda env，避免依赖冲突。详见各方法子目录 README 与 `scripts/`。

## 操作手册

[`docs/howto-reproduce.md`](docs/howto-reproduce.md) — 从工具链安装到单方法 walkthrough 的全流程。
