"""
Shared evaluation utilities.

This module defines all invariants shared across evaluation modes:
- config loading
- dataset loading
- model reconstruction
- checkpoint loading
- tokenizer loading

If something changes here, all evaluations change together.
"""

import sys
from pathlib import Path
import importlib
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

# --------------------------------------------------
# Project root bootstrap
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Config loading
# --------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_frozen_configs(exp_dir: Path) -> tuple[dict, dict]:
    frozen_cfg = load_yaml(exp_dir / "frozen_configs.yaml")
    return frozen_cfg["model"], frozen_cfg["data"]


# --------------------------------------------------
# Dataset loading
# --------------------------------------------------

def load_dataset(data_cfg: dict, model_name: str, split: str):
    processed_dir = Path(
        data_cfg["dataset"]["processed_dir"] + f"_{model_name}"
    )
    return load_from_disk(processed_dir / split)


# --------------------------------------------------
# Model reconstruction
# --------------------------------------------------

def build_model(model_cfg: dict):
    variant = model_cfg["variant"]
    base_model = model_cfg["base_model"]

    module_path = f"models.{variant}.build"
    module = importlib.import_module(module_path)
    return module.build(base_model)


def load_model_and_tokenizer(model_cfg: dict, checkpoint_dir: Path):
    model = build_model(model_cfg)

    model = model.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.float32,
    ).to(device)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"])
    return model, tokenizer
