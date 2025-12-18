"""
Quantitative (metric-based) evaluation.

This module performs evaluation using HuggingFace Trainer and produces
scalar metrics only. No generation or decoding occurs here.
"""

import math
from transformers import Trainer, TrainingArguments


def evaluate_metrics(
    model,
    dataset,
    output_dir,
    per_device_eval_batch_size: int = 1,
) -> dict:
    """
    Run metric-only evaluation.

    Returns a dictionary of metrics including perplexity.
    """

    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=per_device_eval_batch_size,
        dataloader_drop_last=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=dataset,
    )

    metrics = trainer.evaluate()
    metrics["perplexity"] = math.exp(metrics["eval_loss"])

    return metrics
