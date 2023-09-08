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


def run_model(models_values, paths_values, experts):
    tokenizer = AutoTokenizer.from_pretrained(
        models_values.model, cache_dir=paths_values.cache_dir
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        models_values.model,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        cache_dir=paths_values.cache_dir
        )

    model = PeftModel.from_pretrained(
        base_model,
        os.path.join(paths_values.output_dir, "general"),
        adapter_name="general",
    )

    embeddings_model = SentenceTransformer("thenlper/gte-small", device="cuda")
    embeddings_tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
    embeddings_max_length = 512
    embeddings = Embeddings(embeddings_model, embeddings_tokenizer, embeddings_max_length)
    router = Router(embeddings, experts)

    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = paths_values.dataset_dir
    os.environ["HF_DATASETS_CACHE"] = paths_values.cache_dir

    # Load dataset from the hub and get a sample
    dataset = load_dataset(models_values.dataset, split="train")
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

        rounter_expert = router.route(sample['instruction'])
        print(f"---- Routing to {rounter_expert}. Ground truth: {expert_name}")

        input_ids = tokenizer(
            prompt, return_tensors="pt", truncation=True
        ).input_ids.cuda()

        model.load_adapter(os.path.join(paths_values.output_dir, expert_name), expert_name)
        model.set_adapter(expert_name)

        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )

        outputs2 = base_model.generate(
            input_ids=input_ids,
            max_new_tokens=500,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
        )

        print(f"Prompt:\n{sample['instruction']}\n")
        print(
            f"Generated instruction (expert):\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
        )
        print(
            f"Generated instruction (base):\n{tokenizer.batch_decode(outputs2.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
        )
        print(f"Ground truth:\n{sample['response']}")
        print("----------------------------------------\n\n")
