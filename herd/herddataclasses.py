from dataclasses import dataclass
from typing import Optional

@dataclass
class LoraConfigValues:
    lora_alpha: int
    lora_dropout: float
    r: int
    bias: str
    task_type: str
    target_modules: Optional[list]

    def __post_init__(self):
        self.lora_alpha = int(self.lora_alpha)
        self.lora_dropout = float(self.lora_dropout)
        self.r = int(self.r)
        if self.target_modules:
            self.target_modules = self.target_modules.split(",")
        else:
            self.target_modules = None

@dataclass
class MoloraConfigValues(LoraConfigValues):
    num_experts: int

    def __post_init__(self):
        super().__post_init__()
        self.num_experts = int(self.num_experts)


@dataclass
class QuantizationConfigValues:
    use_quantization: bool
    load_in_4bit: bool
    bnb_4bit_use_double_quant: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str

    def __post_init__(self):
        self.use_quantization = bool(self.use_quantization)
        self.load_in_4bit = bool(self.load_in_4bit)
        self.bnb_4bit_use_double_quant = bool(self.bnb_4bit_use_double_quant)


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
