import os

import torch
from peft import PeftModel, PeftConfig, LoraModel
from transformers import (
    AutoModelForCausalLM,
)
import matplotlib.pyplot as plt

# def inspect_lora(
#     output_dir: str,
#     cache_dir: str,
# ):
#     general_model: LoraModel = get_model_for(output_dir, cache_dir, "general")
#     qa_model: LoraModel = get_model_for(output_dir, cache_dir, "qa")
#     print(general_model)

#     # #iterate over all layers and get the ones starting with 'lora_'
#     # for name, param in general_model.named_parameters():
#     #     if "lora_" in name:
#     #         print(name, param.data)

#     general_model.disable_adapter_layers()
#     qa_model.disable_adapter_layers()

#     w1 = general_model.base_model.model.model.layers[31].self_attn.v_proj.lora_B.default.weight
#     w2 = qa_model.base_model.model.model.layers[31].self_attn.v_proj.lora_B.default.weight

#     print(torch.allclose(w1, w2))


# def get_model_for(output_dir, cache_dir, adapter_name):
#     config = PeftConfig.from_pretrained(os.path.join(output_dir, "lora", adapter_name))
#     model = AutoModelForCausalLM.from_pretrained(
#         config.base_model_name_or_path,
#         torch_dtype=torch.float,
#         device_map="cpu",
#         cache_dir=cache_dir,
#     )
#     model = PeftModel.from_pretrained(model, os.path.join(output_dir, "lora", adapter_name))
#     return model


def inspect_lora(output_dir: str, cache_dir: str, adapter_path: str):
    # load pytorch weights
    base_weights = torch.load(os.path.join(output_dir, adapter_path, 'adapter_model.bin'))
    print(type(base_weights))
    print(base_weights.keys())

    # probably want to inspect lora_A and lora_B weights
    lora_weights = base_weights['base_model.model.model.layers.31.self_attn.v_proj.lora_B']



if __name__ == "__main__":
    inspect_lora(
        output_dir="/work3/s212722/herd/meta-llama/Llama-2-7b-hf",
        cache_dir="/work3/s212722/herd/cache",
        adapter_path="alpaca-10/q_v_molora/molora/base",
    )