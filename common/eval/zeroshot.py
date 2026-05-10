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
