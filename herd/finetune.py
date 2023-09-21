import os
from configparser import ConfigParser
from dataclasses import asdict, dataclass
from typing import Dict

import datasets
import torch
from loguru import logger
from herd.models import ModelValues, PathValues
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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


def finetune(
    model_values: ModelValues,
    path_values: PathValues,
    config: ConfigParser,
    experts: Dict,
) -> None:
    dataset = datasets.load_dataset(model_values.dataset, split="train")

    logger.info(
        f"model_id: {model_values.model}, base_dir: {path_values.base_dir}, dataset_dir: {path_values.dataset_dir}, output_dir: {path_values.output_dir}, dataset: {model_values.dataset}",
    )

    # BitsAndBytesConfig int-4 config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_values.model,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        cache_dir=path_values.cache_dir,
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_values.model, cache_dir=path_values.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(**asdict(LoraConfigValues(**dict(config.items("LoraConfig")))))

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    for expert_name, expert_data in experts.items():
        os.environ["WANDB_NAME"] = expert_name

        expert_dataset = dataset.filter(lambda row: row["category"] in expert_data["categories"])

        training_args = TrainingArguments(
            **asdict(TrainingArgumentsValues(**dict(config.items("TrainingArguments"))))
        )

        # set output dir to contain expert name
        training_args.output_dir = os.path.join(training_args.output_dir, expert_name)

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

        logger.info(f"Training expert: {expert_name}, output_dir: {training_args.output_dir}")

        # train
        trainer.train()

        # save model
        trainer.save_model(training_args.output_dir)


@dataclass
class LoraConfigValues:
    lora_alpha: int
    lora_dropout: float
    r: int
    bias: str
    task_type: str

    def __post_init__(self):
        self.lora_alpha = int(self.lora_alpha)
        self.lora_dropout = float(self.lora_dropout)
        self.r = int(self.r)


@dataclass
class TrainingArgumentsValues:
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    optim: str
    logging_steps: int
    save_strategy: str
    learning_rate: float
    bf16: bool
    tf32: bool
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    output_dir: str
    report_to: str

    def __post_init__(self):
        self.num_train_epochs = int(self.num_train_epochs)
        self.per_device_train_batch_size = int(self.per_device_train_batch_size)
        self.gradient_accumulation_steps = int(self.gradient_accumulation_steps)
        self.logging_steps = int(self.logging_steps)
        self.bf16 = bool(self.bf16)
        self.tf32 = bool(self.tf32)
        self.learning_rate = float(self.learning_rate)
        self.max_grad_norm = float(self.max_grad_norm)
        self.warmup_ratio = float(self.warmup_ratio)
