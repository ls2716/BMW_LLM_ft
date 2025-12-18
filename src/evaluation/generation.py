"""
Qualitative (generation-based) evaluation.

This module performs deterministic generation from truncated prompts
and returns decoded text samples suitable for human inspection.
"""

import torch

PROMPT_LEN = 32


def extract_prompt(input_ids):
    return input_ids[:PROMPT_LEN]


def generate_samples(
    dataset,
    model,
    tokenizer,
    num_samples: int,
    max_new_tokens: int = 96,
):
    """
    Generate text samples from the evaluation dataset.

    Returns a list of dictionaries ready for JSONL serialization.
    """

    samples = []

    for i, ex in enumerate(dataset):
        prompt_ids = extract_prompt(ex["input_ids"])

        input_ids = torch.tensor(
            prompt_ids, dtype=torch.long
        ).unsqueeze(0).to(model.device)

        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
            )

        samples.append(
            {
                "sample_id": i,
                "prompt_tokens": PROMPT_LEN,
                "prompt_text": tokenizer.decode(
                    prompt_ids, skip_special_tokens=True
                ),
                "generated_text": tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                ),
                "true_text": tokenizer.decode(
                    ex["input_ids"], skip_special_tokens=True
                ),
            }
        )

        if len(samples) >= num_samples:
            break

    return samples