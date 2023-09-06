import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
from random import randrange
import os


def run_model(models_values, paths_values):
    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        paths_values.output_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        cache_dir=paths_values.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        paths_values.output_dir, cache_dir=paths_values.cache_dir
    )

    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = paths_values.dataset_dir
    os.environ["HF_DATASETS_CACHE"] = paths_values.cache_dir

    # Load dataset from the hub and get a sample
    dataset = load_dataset(models_values.dataset, split="train")
    for i in range(5):
        sample = dataset[randrange(len(dataset))]

        prompt = f"""### Instruction:
    Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

    ### Input:
    {sample['response']}

    ### Response:
    """

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )

        print(f"Prompt:\n{sample['response']}\n")
        print(
            f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
        )
        print(f"Ground truth:\n{sample['instruction']}")
