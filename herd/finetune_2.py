import os
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from typing import Dict

import datasets
import torch
from loguru import logger
from herd.models import ModelValues, PathValues
from herd.herddataclasses import (
    LoraConfigValues,
    MoloraConfigValues,
    QuantizationConfigValues,
    TrainingArgumentsValues,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, IA3Config #, MoloraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def format_instruction(sample: dict) -> str:
    return f"""{sample['system']}

### Input:
{sample['instruction']}

### Response:
{sample['response']}
"""

def finetune2(
    model_values: ModelValues,
    path_values: PathValues,
    config: ConfigParser,
    experts: Dict,
    peft_strategy: str = 'lora',
    is_base = False,
    expert_name = None,
) -> None:

    if not is_base and expert_name is None:
        expert_name = "default"

    dataset = datasets.load_dataset(model_values.dataset, split="train")

    logger.info(
        f"model_id: {model_values.model}, base_dir: {path_values.base_dir}, dataset_dir: {path_values.dataset_dir}, output_dir: {path_values.output_dir}, dataset: {model_values.dataset}",
    )

    if peft_strategy == "lora":
        logger.info("Fine-tuning with LoRA")
        peft_config = LoraConfig(**asdict(LoraConfigValues(**dict(config.items("LoraConfig")))))
            # BitsAndBytesConfig int-4 config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    # elif peft_strategy == "molora":
    #     logger.info("Fine-tuning with MoLoRA")
    #     peft_config = MoloraConfig(**asdict(LoraConfigValues(**dict(config.items("LoraConfig")))))
    #         # BitsAndBytesConfig int-4 config
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_use_double_quant=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #     )
    elif peft_strategy == "ia3":
        logger.info("Fine-tuning with IA3")
        peft_config = IA3Config(task_type="CAUSAL_LM")
        #  IA3 only supports 8bit quantization
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(f"Unknown peft_strategy: {peft_strategy}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_values.model,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        cache_dir=path_values.cache_dir,
    )
    model = prepare_model_for_kbit_training(model)

    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(model_values.model, cache_dir=path_values.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # prepare model for training
    model = get_peft_model(model, peft_config, "base")

    training_args = TrainingArguments(
        **asdict(TrainingArgumentsValues(**dict(config.items("TrainingArguments"))))
    )
    if not os.path.join(training_args.output_dir, peft_strategy):
        os.path.join(training_args.output_dir, peft_strategy)

    if is_base:
        finetune_base(model, dataset, tokenizer, training_args, peft_config, peft_strategy)
    else:
        # load adapter as base for new adapters
        model.load_adapter(os.path.join(path_values.output_dir, peft_strategy, "base"), "base")
        finetune_expert(model, dataset, tokenizer, training_args, peft_config, peft_strategy, experts, expert_name)


def finetune_base(model, dataset, tokenizer, training_args, peft_config, peft_strategy):
    training_args.output_dir = os.path.join(training_args.output_dir, peft_strategy, "base")
    logger.info(f"Pretraining model, output_dir: {training_args.output_dir}")
    os.environ["WANDB_NAME"] = "base"

    # pretrain experts on 10% of the data and use weighs as a base for fine-tuning
    pretrain_dataset = dataset.select(range(len(dataset) // 10))

    trainer = SFTTrainer(
        model=model,
        train_dataset=pretrain_dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=training_args,
    )

    # train
    trainer.train()
    trainer.save_model(training_args.output_dir)

def finetune_expert(model, dataset, tokenizer, training_args, peft_config, peft_strategy, experts, expert_name):
    training_args.output_dir = os.path.join(training_args.output_dir, peft_strategy, expert_name)
    logger.info(f"Pretraining model, output_dir: {training_args.output_dir}")
    os.environ["WANDB_NAME"] = expert_name

    expert_dataset = dataset.filter(lambda row: row["category"] in experts[expert_name]["categories"])

    trainer = SFTTrainer(
        model=model,
        train_dataset=expert_dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_instruction,
        args=training_args,
    )

    # train
    trainer.train()
    trainer.save_model(training_args.output_dir)
