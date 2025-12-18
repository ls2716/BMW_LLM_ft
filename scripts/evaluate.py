"""
Evaluation entry point script.
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

import argparse
import json
import yaml
from datetime import datetime

from src.evaluation.common import (
    load_frozen_configs,
    load_dataset,
    load_model_and_tokenizer,
)
from src.evaluation.metrics import evaluate_metrics
from src.evaluation.generation import generate_samples


# --------------------------------------------------
# Main
# --------------------------------------------------

def main(
    experiment: str,
    checkpoint: str,
    split: str,
    mode: str,
    num_samples: int,
):
    experiments_dir = PROJECT_ROOT / "experiments"
    exp_dir = experiments_dir / experiment

    if not exp_dir.exists():
        raise RuntimeError(f"Experiment not found: {exp_dir}")

    # --------------------------------------------------
    # Load frozen configs
    # --------------------------------------------------
    model_cfg, data_cfg = load_frozen_configs(exp_dir)
    model_name = model_cfg["base_model"].split("/")[-1]

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------
    dataset = load_dataset(data_cfg, model_name, split)

    # --------------------------------------------------
    # Load model + tokenizer
    # --------------------------------------------------
    ckpt_dir = exp_dir / "checkpoints" / checkpoint
    if not ckpt_dir.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_dir}")


    model, tokenizer = load_model_and_tokenizer(model_cfg, ckpt_dir)

    # --------------------------------------------------
    # Evaluation root directory
    # --------------------------------------------------
    evals_root = PROJECT_ROOT / "evaluations"
    eval_name = f"eval_{experiment}_{checkpoint}_{split}"
    eval_dir = evals_root / eval_name
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Starting evaluation: {eval_name}")
    print(f"[INFO] Mode: {mode}")

    # --------------------------------------------------
    # Metrics evaluation
    # --------------------------------------------------
    if mode in {"metrics", "all"}:
        print("[INFO] Running metric evaluation")

        metrics = evaluate_metrics(
            model=model,
            dataset=dataset,
            output_dir=eval_dir / "tmp_metrics",
        )

        with open(eval_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # --------------------------------------------------
    # Generation evaluation
    # --------------------------------------------------
    if mode in {"generation", "all"}:
        print("[INFO] Running generation evaluation")

        samples = generate_samples(
            dataset=dataset,
            model=model,
            tokenizer=tokenizer,
            num_samples=num_samples,
        )

        with open(eval_dir / "generated_samples.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")


    # --------------------------------------------------
    # Write evaluation metadata (immutable)
    # --------------------------------------------------
    meta = {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "split": split,
        "mode": mode,
        "num_samples": num_samples if mode in {"generation", "all"} else None,
        "timestamp": datetime.now().isoformat(),
    }

    with open(eval_dir / "eval_meta.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    with open(eval_dir / "checkpoint_used.txt", "w") as f:
        f.write(str(ckpt_dir))

    print(f"[INFO] Evaluation complete: {eval_name}")


# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", default="final")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--mode",
        choices=["metrics", "generation", "all"],
        default="all",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of generation samples (generation/all modes only)",
    )

    args = parser.parse_args()

    main(
        experiment=args.experiment,
        checkpoint=args.checkpoint,
        split=args.split,
        mode=args.mode,
        num_samples=args.num_samples,
    )
