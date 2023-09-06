import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from loguru import logger

from transformers import BitsAndBytesConfig
import torch

from configparser import ConfigParser
from dataclasses import dataclass, asdict


def format_instruction(sample: dict) -> str:
    return f"""### Instruction:
Use the Input below to create an instruction, which could have been used to generate the input using an LLM.

### Input:
{sample['response']}

### Response:
{sample['instruction']}
"""


def finetune(models_values, paths_values, config: ConfigParser) -> None:
    dataset = datasets.load_dataset(models_values.dataset, split="train")

    logger.info(
        f"model_id: {models_values.model}, base_dir: {paths_values.base_dir}, dataset_dir: {paths_values.dataset_dir}, output_dir: {paths_values.output_dir}, dataset: {models_values.dataset}",
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
        models_values.model,
        quantization_config=bnb_config,
        use_cache=False,
        device_map="auto",
        cache_dir=paths_values.cache_dir,
    )
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(
        models_values.model, cache_dir=paths_values.cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config based on QLoRA paper
    peft_config = LoraConfig(
        **asdict(LoraConfigValues(**dict(config.items("LoraConfig"))))
    )

    # prepare model for training
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        **asdict(TrainingArgumentsValues(**dict(config.items("TrainingArguments"))))
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
