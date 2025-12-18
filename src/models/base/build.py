from transformers import AutoModelForCausalLM


def build(base_model_id):
    return AutoModelForCausalLM.from_pretrained(base_model_id)