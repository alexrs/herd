import os

import datasets
import torch
from peft import PeftModel, PeftConfig
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
):
    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = dataset_dir

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", cache_dir=cache_dir
    )

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    config = PeftConfig.from_pretrained(os.path.join(output_dir, peft_strategy, "general"))

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        torch_dtype=torch.float,
        quantization_config=quantization_config,
        # device_map="cpu",
        cache_dir=cache_dir,
    )

    model = PeftModel.from_pretrained(model, os.path.join(output_dir, peft_strategy, "general"))
    model.load_adapter(os.path.join(output_dir, peft_strategy, "code"), "code")
    model.load_adapter(os.path.join(output_dir, peft_strategy, "creative"), "creative")

    model.add_weighted_adapter(["code", "creative"], [0.5, 0.5], "code_creative")
    model.set_adapter("code_creative")

    prompt = f"""
    ### Input:
    What is the capital of Spain?

    ### Response:
    """

    input_ids = tokenizer(
        prompt, return_tensors="pt", truncation=True
    ).input_ids.cuda()

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


if __name__ == "__main__":
    run_model(
        output_dir="/work3/s212722/herd/meta-llama/Llama-2-7b-hf",
        cache_dir="/work3/s212722/herd/cache",
        dataset_dir="/work3/s212722/herd/datasets",
        peft_strategy = 'ia3',
    )