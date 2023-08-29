import argparse
import os
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
import configparser


def format_instruction(sample):
    return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['response']}

### Response:
{sample['instruction']}
"""


def extract_variables(args):
    model_id = args.model
    base_dir = args.base_dir
    dataset_dir = os.path.join(base_dir, "datasets/")
    cache_dir = os.path.join(dataset_dir, "cache")
    output_dir = os.path.join(base_dir, model_id)
    config_file = args.config_file
    dataset = args.dataset

    return model_id, base_dir, dataset_dir, cache_dir, output_dir, config_file, dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Llama 2 models")
    parser.add_argument("--model", default="meta-llama/Llama-2-7b")
    parser.add_argument("--base_dir", default="/work3/s212722/")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--config_file", default="config.ini")

    (
        model_id,
        _,
        dataset_dir,
        cache_dir,
        output_dir,
        config_file,
        dataset,
    ) = extract_variables(parser.parse_args())

    # Load dataset from the hub
    datasets.config.DOWNLOADED_DATASETS_PATH = dataset_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir

    dataset = datasets.load_dataset(dataset, split="train")

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        cache_dir=cache_dir,
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    config = configparser.ConfigParser()
    config.read(config_file)

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        lora_alpha=config.getint("LoraConfig", "lora_alpha", fallback=16),
        lora_dropout=config.getfloat("LoraConfig", "lora_dropout", fallback=0.1),
        r=config.getint("LoraConfig", "r", fallback=64),
        bias=config.get("LoraConfig", "bias", fallback="none"),
        task_type=config.get("LoraConfig", "task_type", fallback="CAUSAL_LM"),
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.getint(
            "TrainingArguments", "num_train_epochs", fallback=3
        ),
        per_device_train_batch_size=config.getint(
            "TrainingArguments", "per_device_train_batch_size", fallback=4
        ),
        gradient_accumulation_steps=config.getint(
            "TrainingArguments", "gradient_accumulation_steps", fallback=2
        ),
        gradient_checkpointing=config.getboolean(
            "TrainingArguments", "gradient_checkpointing", fallback=True
        ),
        optim=config.get("TrainingArguments", "optim", fallback="paged_adamw_32bit"),
        logging_steps=config.getint("TrainingArguments", "logging_steps", fallback=10),
        save_strategy=config.get(
            "TrainingArguments", "save_strategy", fallback="epoch"
        ),
        learning_rate=config.getfloat(
            "TrainingArguments", "learning_rate", fallback=2e-4
        ),
        bf16=config.getboolean("TrainingArguments", "bf16", fallback=True),
        tf32=config.getboolean("TrainingArguments", "tf32", fallback=True),
        max_grad_norm=config.getfloat(
            "TrainingArguments", "max_grad_norm", fallback=0.3
        ),
        warmup_ratio=config.getfloat(
            "TrainingArguments", "lr_scheduler_type", fallback=0.03
        ),
        lr_scheduler_type=config.get(
            "TrainingArguments", "lr_scheduler_type", fallback="constant"
        ),
        disable_tqdm=config.get(
            "TrainingArguments", "disable_tqdm", fallback=True
        ),  # disable tqdm since with packing values are in correct
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=training_args,
    )

    # train
    trainer.train()  # there will not be a progress bar since tqdm is disabled

    # save model
    trainer.save_model()


if __name__ == "__main__":
    main()
