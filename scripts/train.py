"""
Training entry point for LLM fine-tuning.
"""

# --------------------------------------------------
# Bootstrap import path (REQUIRED)
# --------------------------------------------------
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# --------------------------------------------------

import re
import yaml
import json
import importlib
import subprocess
from datetime import datetime

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# --------------------------------------------------
# Config loading
# --------------------------------------------------


def load_yaml(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_configs():
    return {
        "data": load_yaml(PROJECT_ROOT / "configs/data.yaml"),
        "model": load_yaml(PROJECT_ROOT / "configs/model.yaml"),
        "training": load_yaml(PROJECT_ROOT / "configs/training.yaml"),
    }


# --------------------------------------------------
# Run ID handling
# --------------------------------------------------


def next_run_id(experiments_dir: Path) -> int:
    experiments_dir.mkdir(exist_ok=True)
    pattern = re.compile(r"exp_(\d+)_")

    ids = []
    for p in experiments_dir.iterdir():
        if p.is_dir():
            m = pattern.match(p.name)
            if m:
                ids.append(int(m.group(1)))

    return max(ids) + 1 if ids else 1


# --------------------------------------------------
# Model construction
# --------------------------------------------------


def build_model(model_cfg: dict):
    variant = model_cfg["variant"]
    base_model = model_cfg["base_model"]

    module_path = f"src.models.{variant}.build"
    print(f"[INFO] Loading model variant: {module_path}")

    module = importlib.import_module(module_path)
    model = module.build(base_model)

    return model


# --------------------------------------------------
# Logging callback
# --------------------------------------------------
from transformers import TrainerCallback
import os


class FileLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            os.makedirs(args.logging_dir, exist_ok=True)
            log_file = os.path.join(args.logging_dir, "trainer.log")
            with open(log_file, "a") as f:
                f.write(f"{state.global_step}\t{logs}\n")


# --------------------------------------------------
# Main training logic
# --------------------------------------------------


def main():
    cfg = load_configs()

    experiments_dir = PROJECT_ROOT / "experiments"
    run_id = next_run_id(experiments_dir)

    variant = cfg["model"]["variant"]
    model_name = cfg["model"]["base_model"].split("/")[-1]
    exp_name = f"exp_{run_id:03d}_{model_name}_{variant}"
    exp_dir = experiments_dir / exp_name

    if exp_dir.exists():
        raise RuntimeError(f"Experiment already exists: {exp_dir}")

    print(f"[INFO] Starting experiment: {exp_name}")
    exp_dir.mkdir(parents=True)

    # --------------------------------------------------
    # Save frozen configs
    # --------------------------------------------------
    frozen_cfg = {
        "data": cfg["data"],
        "model": cfg["model"],
        "training": cfg["training"],
    }

    with open(exp_dir / "frozen_configs.yaml", "w") as f:
        yaml.safe_dump(frozen_cfg, f)

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------
    processed_dir = Path(cfg["data"]["dataset"]["processed_dir"] + f"_{model_name}")

    train_ds = load_from_disk(processed_dir / "train")
    val_ds = load_from_disk(processed_dir / "validation")

    print(f"[INFO] Train dataset size: {len(train_ds)}")
    print(f"[INFO] Validation dataset size: {len(val_ds)}")

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------
    model = build_model(cfg["model"])

    # --------------------------------------------------
    # Training arguments
    # --------------------------------------------------
    train_cfg = cfg["training"]

    training_args = TrainingArguments(
        output_dir=str(exp_dir / "checkpoints"),
        overwrite_output_dir=False,
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=train_cfg["weight_decay"],
        num_train_epochs=train_cfg["num_train_epochs"],
        warmup_steps=train_cfg["warmup_steps"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        logging_steps=train_cfg["logging_steps"],
        logging_dir=exp_dir / "logs",
        eval_strategy=train_cfg["eval_strategy"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=2,
        fp16=train_cfg["fp16"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to=train_cfg["report_to"],
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.add_callback(FileLoggerCallback)

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    trainer.train()

    # --------------------------------------------------
    # Save final model
    # --------------------------------------------------
    trainer.save_model(exp_dir / "checkpoints" / "final")

    print(f"[INFO] Training complete: {exp_name}")


if __name__ == "__main__":

    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    print(f"[INFO] GPU count: {torch.cuda.device_count()}")
    
    main()
