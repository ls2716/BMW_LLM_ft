"""
Config-driven preprocessing for causal LLM fine-tuning.

Input:
- Raw CSV (from data.yaml)
- Base tokenizer (unchanged)

Output:
- Hugging Face Arrow datasets:
  input_ids, attention_mask, labels
"""

import json
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from transformers import AutoTokenizer


# -------------------------
# Utilities
# -------------------------

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def pack_sequences(token_ids, max_length, pad_token_id, eos_token_id):
    sequences = []
    current = []

    for tid in token_ids:
        current.append(tid)
        if len(current) == max_length:
            sequences.append(current)
            current = []

    if current:
        current.append(eos_token_id)
        current = current[:max_length]
        pad_len = max_length - len(current)
        current += [pad_token_id] * pad_len
        sequences.append(current)

    return sequences


# -------------------------
# Main
# -------------------------

def main():
    cfg = load_config("configs/data.yaml")
    model_cfg = load_config("configs/model.yaml")
    model_name = model_cfg["base_model"]
    tok_cfg = cfg["tokenizer"]["source"] = model_name

    raw_csv = Path(cfg["dataset"]["raw_csv"])
    processed_dir = Path(cfg["dataset"]["processed_dir"] + f"_{model_name.split('/')[-1]}")
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits_cfg = cfg["splits"]
    seq_cfg = cfg["sequence"]
    tok_cfg = cfg["tokenizer"]

    np.random.seed(splits_cfg["seed"])

    # Load raw data
    df = pd.read_csv(raw_csv)
    assert "content" in df.columns, "CSV must contain a 'content' column"

    df = df.sample(frac=1.0, random_state=splits_cfg["seed"]).reset_index(drop=True)

    n = len(df)
    n_test = int(n * splits_cfg["test_ratio"])
    n_val = int(n * splits_cfg["validation_ratio"])

    splits = {
        "test": df.iloc[:n_test],
        "validation": df.iloc[n_test : n_test + n_val],
        "train": df.iloc[n_test + n_val :],
    }

    tokenizer = AutoTokenizer.from_pretrained(
        tok_cfg["source"],
        use_fast=tok_cfg["use_fast"],
    )

    pad_token_id = (
        seq_cfg["pad_token_id"]
        if seq_cfg["pad_token_id"] is not None
        else tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    eos_token_id = tokenizer.eos_token_id
    max_length = seq_cfg["max_length"]
    label_pad_id = seq_cfg["label_pad_token_id"]

    for split_name, split_df in splits.items():
        print(f"Preprocessing {split_name} ({len(split_df)} rows)")

        token_stream = []

        for _, row in split_df.iterrows():
            text = row["content"]
            ids = tokenizer(
                text,
                add_special_tokens=tok_cfg["add_special_tokens"],
                truncation=False,
            )["input_ids"]
            token_stream.extend(ids)

        sequences = pack_sequences(
            token_stream,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        examples = []
        for seq in sequences:
            attention_mask = [1 if t != pad_token_id else 0 for t in seq]
            labels = [
                t if t != pad_token_id else label_pad_id
                for t in seq
            ]
            examples.append(
                {
                    "input_ids": seq,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

        dataset = Dataset.from_list(examples)
        split_dir = processed_dir / split_name
        dataset.save_to_disk(split_dir)

        metadata = {
            "split": split_name,
            "num_sequences": len(dataset),
            "tokenizer_source": tok_cfg["source"],
            "max_length": max_length,
            "packing": seq_cfg["packing"],
            "pad_token_id": pad_token_id,
            "label_pad_token_id": label_pad_id,
        }

        with open(split_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    print("Preprocessing complete.")


if __name__ == "__main__":
    main()
