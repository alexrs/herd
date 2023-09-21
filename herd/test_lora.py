import os

import datasets
import torch
from peft import PeftModel, PeftConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

def run_model(
    output_dir: str,
    cache_dir: str,
    dataset_dir: str,
    peft_strategy: str,
    adapter_name: str,
):
    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = dataset_dir

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", cache_dir=cache_dir
    )

    config = PeftConfig.from_pretrained(os.path.join(output_dir, peft_strategy, adapter_name))

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=None,
        device_map="cpu",
        cache_dir=cache_dir,
    )

    model = get_peft_model(model, config)
    model.cuda()

    model.load_adapter(os.path.join(output_dir, peft_strategy, adapter_name), adapter_name)
    model.load_adapter(os.path.join(output_dir, peft_strategy, "expert2"),  "expert2")
    model.load_adapter(os.path.join(output_dir, peft_strategy, "expert1lora"), "expert1lora")
    # model.load_adapter(os.path.join(output_dir, peft_strategy, "expert2lora"), "expert2lora")

    prompt = f"""
    ### Input:
    What is the capital of Spain?

    ### Response:
    """

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).input_ids.cuda()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output
        return hook

    # print(model)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    print(
        f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )

    model.set_adapter(adapter_name)
    print(model.active_adapters)
    print(model.active_adapter)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    print(
        f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )

    # model.disable_adapters()
    model.set_adapter("expert2")
    print(model.active_adapters)
    print(model.active_adapter)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    print(
        f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )

    # model.disable_adapters()
    model.set_adapter("expert1lora")
    print(model.active_adapter)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    print(
        f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )

    # model.disable_adapters()
    model.set_adapter("expert1")
    model.set_adapter("expert2")
    print(model.active_adapter)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    print(
        f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )

    # model.disable_adapters()
    model.add_weighted_adapter(["expert1", "expert2"], [0.5, 0.5], "expert12")
    model.set_adapter("expert12")
    print(model.active_adapter)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )
    print(
        f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )

    # model.disable_adapters()
    # model.set_adapter("expert2lora")
    # print(model.active_adapters)
    # outputs = model.generate(
    #     input_ids=input_ids,
    #     max_new_tokens=500,
    #     do_sample=True,
    #     top_p=0.9,
    #     temperature=0.9,
    # )
    # print(
    #     f"Response: \n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    # )


if __name__ == "__main__":
    run_model(
        output_dir="/work3/s212722/herd/meta-llama/Llama-2-7b-hf",
        cache_dir="/work3/s212722/herd/cache",
        dataset_dir="/work3/s212722/herd/datasets",
        peft_strategy = 'lora',
        adapter_name= "expert1"
    )