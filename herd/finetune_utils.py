from dataclasses import asdict
from configparser import ConfigParser
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from herd.herddataclasses import (
    LoraConfigValues,
    MoloraConfigValues,
    QuantizationConfigValues,
)
from peft import LoraConfig, prepare_model_for_kbit_training, IA3Config, MoloraConfig
from loguru import logger

def quantization_config(config: ConfigParser) -> BitsAndBytesConfig:
    quantization_config = QuantizationConfigValues(**dict(config.items("QuantizationConfig")))
    if quantization_config.use_quantization:
        # delete use_quantization from config
        quantization_config_dict = asdict(quantization_config)
        quantization_config_dict.pop("use_quantization")
        return BitsAndBytesConfig(**quantization_config_dict)
    return None

def get_peft_config(config, peft_strategy):
    if peft_strategy == "lora":
        logger.info("Fine-tuning with LoRA")
        return LoraConfig(**asdict(LoraConfigValues(**dict(config.items("LoraConfig")))))
    elif peft_strategy == "ia3":
        logger.info("Fine-tuning with IA3")
        return IA3Config(task_type="CAUSAL_LM")
    elif peft_strategy == "molora":
        peft_config = MoloraConfig(**asdict(MoloraConfigValues(**dict(config.items("MoLoraConfig")))))
        logger.info(f"Fine-tuning with MoLoRA using {peft_config.num_experts} experts")
        return peft_config
    else:
        raise ValueError(f"Unknown peft_strategy: {peft_strategy}")

def prepare_model(pretrained_model_name_or_path, cache_dir, config, peft_strategy):
    # Load peft config
    peft_config = get_peft_config(config, peft_strategy)
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path,
        quantization_config=quantization_config(config),
        device_map="auto",
        cache_dir=cache_dir,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.pretraining_tp = 1
    return model, peft_config


def prepare_tokenizer(pretrained_model_name_or_path, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer
