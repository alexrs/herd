import os
from random import randrange
from typing import Dict

import datasets
import torch
import torch.nn as nn
from datasets import load_dataset
from herd.embeddings import Embeddings
from peft import PeftModel
from herd.models import ModelValues, PathValues
from herd.router import Router
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from loguru import logger
import datetime
import numpy as np

def route_to_experts(model, router, instruction: str, top: int = 1):
    # Disable all adapters
    model.disable_adapter()
    logger.debug(f"Routing to {top} experts")
    # Experts is a list of tuples (expert_name, score).
    experts = router.route(instruction, top)
    if top == 1:
        # If we only want the top expert, set it as an adapter
        logger.debug(f"Setting adapter to {experts[0][0]}")
        model.set_adapter(experts[0][0])
    else:
        # Otherwise, we compute a new adapter as a combination of the top experts.
        # We generate a unique name for the adapter because even if the same experts are used
        # the weights may be different.
        adapter_name = str(hash(datetime.datetime.now()))

        weights = np.array([expert[1] for expert in experts])
        inverted_weights = 1 / weights
        # w = inverted_weights / np.sum(inverted_weights)
        # use softmax for the weights
        w = nn.functional.softmax(torch.tensor(inverted_weights), dim=0).numpy()
        e = [expert[0] for expert in experts]
        logger.debug(f"Creating adapter for: {list(zip(e, w))}")

        # TODO: Experiment with other routing methods
        # TODO: There is not add_weighted_adapter in IA3Model. Can we add it? - Yes, but it's not used in the code.
        model.add_weighted_adapter(
            e,
            w,
            combination_type="linear",
            adapter_name=adapter_name,
        )
        model.set_adapter(adapter_name)

    return experts


# TODO: add ia3
# TODO: add quantization use optional
def run_model(
    model_values: ModelValues,
    path_values: PathValues,
    experts: Dict,
    only_base: bool = False,
    interactive: bool = False,
    peft_strategy: str = 'lora',
    top: int = 1,
):
    tokenizer = AutoTokenizer.from_pretrained(
        model_values.model, cache_dir=path_values.cache_dir
    )

    if peft_strategy == "lora" or peft_strategy == "molora":
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
    else:
        raise ValueError(f"Unknown peft_strategy: {peft_strategy}")

    if not only_base:
        model = PeftModel.from_pretrained(
            model,
            os.path.join(path_values.output_dir, peft_strategy, "general"),
            adapter_name="general",
        )

        print(model)

        # Load adapters
        for expert_name in experts.keys():
            model.load_adapter(
                os.path.join(path_values.output_dir, peft_strategy, expert_name), expert_name
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
                route_to_experts(model, router, instruction, top)

            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
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
