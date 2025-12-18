"""
Qualitative question-answering evaluation.

This module performs deterministic generation from truncated prompts
and returns decoded text samples suitable for human inspection.
"""

import torch


def generate_samples(
    qa_dataset,
    model,
    tokenizer,
    num_samples: int,
    max_new_tokens: int = 96,
):
    """
    Generate text samples from the evaluation dataset.

    Returns a list of dictionaries ready for JSONL serialization.
    """

    answer_samples = []

    for i, ex in enumerate(qa_dataset):
        question = ex["question"]
        true_answer = ex["answer"]
        prompt_text = f"Q: {question}\nA:"
        prompt_ids = tokenizer.encode(
            prompt_text, return_tensors="pt"
        ).squeeze(0)
        input_ids = prompt_ids.unsqueeze(0).to(model.device)

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

        answer_samples.append(
            {
                "sample_id": i,
                "prompt_text": prompt_text,
                "generated_answer": tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
                ),
                "true_answer": true_answer,
            }
        )
    
    return answer_samples
        