import os

import datasets
import torch
from peft import PeftModel, PeftConfig, MoloraConfig, get_peft_model, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from prompter import Prompter

import math

def run_model(
    output_dir: str,
    cache_dir: str,
    dataset_dir: str,
    peft_strategy: str,
    adapter_name: str,
):
    # Load dataset from the hub
    # datasets.config.DOWNLOADED_DATASETS_PATH = dataset_dir

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", cache_dir=cache_dir
    )

    # config = MoloraConfig.from_pretrained(os.path.join(output_dir, 'alpaca-5', peft_strategy, 'router'))
    # config.experts_to_combine = ["expert_1", "expert_2"]
    # config.num_experts = 2
    # config.inference_mode = False
    # config.only_router = True

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoPeftModelForCausalLM.from_pretrained(
        os.path.join(output_dir, 'router_top_2'),
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
    )

    # model_peft = get_peft_model(model, config)
    # model_peft.print_trainable_parameters()

    prompter = Prompter()
    while True:
        inp = input()
        prompt = prompter.generate_prompt({'instruction': inp, 'input': None, 'output': None})

        input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )
        print(f"Response: \n{tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}")


    from peft.utils import load_peft_weights
    ww = load_peft_weights(os.path.join(output_dir, 'alpaca-5', peft_strategy, 'expert_1'))
    ww2 = load_peft_weights(os.path.join(output_dir, 'alpaca-5', peft_strategy, 'expert_2'))

    w0 = model_peft.base_model.model.model.layers[3].self_attn.v_proj.lora_A.default[0:1]
    print(f"Peft Model 0: {w0[0]}")
    w01 = model_peft.base_model.model.model.layers[3].self_attn.v_proj.lora_A.default[1:2]
    print(f"Peft Model 0: {w01[0]}")
    w1 = ww['base_model.model.model.layers.3.self_attn.v_proj.lora_A']
    print(f"Expert 1: {w1[0]}")
    w3 = ww2['base_model.model.model.layers.3.self_attn.v_proj.lora_A']
    print(f" Expert 2: {w3[0]}")

    model_peft.load_experts(os.path.join(output_dir, 'alpaca-5', peft_strategy), 'default', ["expert_1", "expert_2"], True)

    w2 = model_peft.base_model.model.model.layers[3].self_attn.v_proj.lora_A.default[0:1]
    print(f"Experts loaded (e1): {w2[0]}")

    w4 = model_peft.base_model.model.model.layers[3].self_attn.v_proj.lora_A.default[1:2]
    print(f"Experts loaded (e2): {w4[0]}")

    wrouter = model_peft.base_model.model.model.layers[3].self_attn.v_proj.lora_router.default.weight
    print(f"Lora Router: {wrouter}")

    def reset_router_weights(model):
        for name, param in model.named_parameters():
            if "lora_router" in name and "weight" in name:
                # use torch.nn.init.kaiming_uniform_
                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))

            if "lora_router" in name and "bias" in name:
                # initialize to zero
                torch.nn.init.zeros_(param)


    reset_router_weights(model_peft)

    wrouter = model_peft.base_model.model.model.layers[3].self_attn.v_proj.lora_router.default.weight
    print(f"Lora Router: {wrouter}")

    print(torch.allclose(w1, w2))

    model_peft.print_trainable_parameters()
    model_peft._mark_only_router_as_trainable()
    for name, param in model_peft.named_parameters():
        if param.requires_grad:
            print(name)

    # prompt = f"""
    # ### Input:
    # What is the capital of Spain?

    # ### Response:
    # """

    # # input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()

    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output
    #     return hook

    # print(model_peft)

    # for name, layer in model_peft.named_modules():
    #     layer.register_forward_hook(get_activation(name))

    # outputs = model_peft.generate(
    #     input_ids=input_ids,
    #     max_new_tokens=500,
    #     do_sample=True,
    #     top_p=0.9,
    #     temperature=0.9,
    # )

    # print(
    #     f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    # )

    # for key in activation:
    #     print(key)
    #     if "lora_router" in key:
    #         print(activation[key])


if __name__ == "__main__":
    run_model(
        output_dir="/work3/s212722/herd/meta-llama/Llama-2-7b-hf/alpaca-5/all_linear_paged_adam/molora/",
        cache_dir="/work3/s212722/herd/cache",
        dataset_dir="/work3/s212722/herd/datasets",
        peft_strategy = 'molora',
        adapter_name= "expert1"
    )