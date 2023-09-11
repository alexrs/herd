import torch
from peft import PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import datasets
from datasets import load_dataset
from random import randrange
import os
from sentence_transformers import SentenceTransformer

from router import Router
from embeddings import Embeddings
from typing import Dict
from models import ModelValues, PathValues


def run_model(
    model_values: ModelValues,
    path_values: PathValues,
    experts: Dict,
    only_base: bool = False,
    interactive: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_values.model, cache_dir=path_values.cache_dir
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_values.model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=path_values.cache_dir,
    )

    if not only_base:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(path_values.output_dir, "general"),
            adapter_name="general",
        )

        # Load adapters
        for expert_name in experts.keys():
            model.load_adapter(
                os.path.join(path_values.output_dir, expert_name), expert_name
            )

    embeddings_model = SentenceTransformer(model_values.embeddings_model, device="cuda")
    embeddings_tokenizer = AutoTokenizer.from_pretrained(model_values.embeddings_model)
    embeddings = Embeddings(
        embeddings_model, embeddings_tokenizer, model_values.embeddings_max_length
    )
    router = Router(embeddings, experts)

    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = path_values.dataset_dir
    os.environ["HF_DATASETS_CACHE"] = path_values.cache_dir

    if interactive:
        while True:
            instruction = input("Enter instruction: ")
            if instruction == "exit":
                break

            if instruction == "":
                continue

            prompt = f"""
            ### Input:
            {instruction}

            ### Response:
            """

            if not only_base:
                rounter_expert = router.route(instruction)
                print(f"---- Routing to {rounter_expert}")
                model.set_adapter(rounter_expert)

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
    else:
        # Load dataset from the hub and get a sample
        dataset = load_dataset(model_values.dataset, split="train")
        for expert_name, expert_data in experts.items():
            expert_dataset = dataset.filter(
                lambda row: row["category"] in expert_data["categories"]
            )
            sample = expert_dataset[randrange(len(expert_dataset))]

            prompt = f"""{sample['system']}

        ### Input:
        {sample['instruction']}

        ### Response:
        """

            rounter_expert = router.route(sample["instruction"])
            print(f"---- Routing to {rounter_expert}. Ground truth: {expert_name}")

            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True
            ).input_ids.cuda()

            model.set_adapter(expert_name)

            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=500,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
            )

            print(f"Prompt:\n{sample['instruction']}\n")
            print(
                f"Generated response (expert):\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
            )
            print(f"Ground truth:\n{sample['response']}")
            print("----------------------------------------\n\n")
