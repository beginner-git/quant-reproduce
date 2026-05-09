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
