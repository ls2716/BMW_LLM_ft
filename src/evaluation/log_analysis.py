from pathlib import Path
import ast
from typing import Dict, List, Any


def parse_trainer_log(log_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse a HuggingFace Trainer log file into structured records.

    Returns a dictionary with keys:
      - "train": list of training step records
      - "eval": list of evaluation records
      - "summary": list of final summary records (usually length 1)

    Each record contains:
      - step (int)
      - epoch (float, if present)
      - remaining logged fields
    """

    train_records = []
    eval_records = []
    summary_records = []

    with open(log_path, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                step_str, payload_str = line.split("\t", 1)
                step = int(step_str)

                payload_str = payload_str.replace("nan", "None")
                payload = ast.literal_eval(payload_str)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse line {line_no}: {raw_line}"
                ) from e

            record = {"step": step, **payload}

            # --------------------------------------------------
            # Classification
            # --------------------------------------------------
            if "loss" in payload and "grad_norm" in payload:
                train_records.append(record)
            elif "eval_loss" in payload:
                eval_records.append(record)
            elif "train_runtime" in payload:
                summary_records.append(record)
            else:
                # Unknown / future-proof bucket
                summary_records.append(record)

    return {
        "train": train_records,
        "eval": eval_records,
        "summary": summary_records,
    }