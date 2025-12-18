from transformers import AutoModelForCausalLM
import torch

def build(base_model_id):
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    layers = model.model.layers
    # Remove the middle layer
    n_layers = len(layers)
    middle_layer = n_layers // 2
    model.model.layers = torch.nn.ModuleList(
        [layer for i, layer in enumerate(layers) if i != middle_layer]
    )
    # Update config
    model.config.num_hidden_layers = len(model.model.layers)

    # Disable cache to avoid potential issues with the modified architecture
    model.config.use_cache = False

    return model