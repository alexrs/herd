import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
from random import randrange
import os
import argparse


def extract_variables(args):
    model_id = args.model
    base_dir = args.base_dir
    dataset_dir = os.path.join(base_dir, "datasets/")
    cache_dir = os.path.join(dataset_dir, "cache")
    output_dir = os.path.join(base_dir, model_id)
    dataset = args.dataset

    return model_id, base_dir, dataset_dir, cache_dir, output_dir, dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Llama 2 models")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--base_dir", default="/work3/s212722/herd/")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")

    (
        _,
        _,
        dataset_dir,
        cache_dir,
        output_dir,
        dataset,
    ) = extract_variables(parser.parse_args())

    # load base LLM model and tokenizer
    model = AutoPeftModelForCausalLM.from_pretrained(
        output_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        cache_dir=cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir, cache_dir=cache_dir)

    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = dataset_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    # Load dataset from the hub and get a sample
    dataset = load_dataset(dataset, split="train")
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


if __name__ == "__main__":
    main()
