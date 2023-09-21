import os
from dataclasses import asdict

import datasets
from loguru import logger
from herd.herddataclasses import TrainingArgumentsValues
from herd.finetune_utils import prepare_model, prepare_tokenizer
from herd.prompter import Prompter

from peft import get_peft_model, AutoPeftModelForCausalLM
from transformers import TrainingArguments
from trl import SFTTrainer
import torch
import math

from peft.utils import load_peft_weights

# def format_instruction(sample: dict) -> str:
#     """
#     Format instruction based on the sample provided.
#     """
#     return f"""{sample['system']}

# ### Input:
# {sample['instruction']}

# ### Response:
# {sample['response']}
# """


def _get_training_arguments(config, peft_strategy, expert_name):
    """
    Get training arguments based on the configuration provided.
    """
    return TrainingArguments(
        **asdict(TrainingArgumentsValues(**dict(config.items("TrainingArguments")))),
        run_name=os.path.join(peft_strategy, expert_name)
    )


def _init_trainer(model_to_train, dataset, peft_config, tokenizer, training_args):
    """
    Initialize and return the SFTTrainer.
    """
    prompter = Prompter()
    return SFTTrainer(
        model=model_to_train,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompter.generate_prompt,
        args=training_args,
    )


def _finetune_base(dataset, config, peft_strategy, tokenizer, model_values, path_values):
    expert_name = "base"
    base_dataset = dataset.shuffle().select(range(len(dataset) // 10))

    training_args = _get_training_arguments(config, peft_strategy, expert_name)
    output_dir = os.path.join(path_values.output_dir, peft_strategy, expert_name)
    training_args.output_dir = output_dir
    # For the base adapter we only wanna train for 1 epoch
    training_args.num_train_epochs = 1

    model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
    if peft_config.num_experts != 1:
        peft_config.num_experts = 1
        logger.warning(f"Number of experts is not 1 but we are training the base expert, setting it to 1. num_experts: {peft_config.num_experts}")

    model_to_train = get_peft_model(model, peft_config)
    model_to_train.print_trainable_parameters()

    trainer = _init_trainer(model_to_train, base_dataset, peft_config, tokenizer, training_args)

    logger.info(f"Training expert: {expert_name}, output_dir: {training_args.output_dir}")
    trainer.train()
    trainer.save_model(training_args.output_dir)


def _finetune_all(dataset, config, peft_strategy, tokenizer, model_values, path_values):
    expert_name = "all"

    training_args = _get_training_arguments(config, peft_strategy, expert_name)
    output_dir = os.path.join(path_values.output_dir, peft_strategy, expert_name)
    training_args.output_dir = output_dir

    model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
    model_to_train = get_peft_model(model, peft_config)
    model_to_train.print_trainable_parameters()


    trainer = _init_trainer(model_to_train, dataset, peft_config, tokenizer, training_args)

    logger.info(f"Training expert: {expert_name}, output_dir: {training_args.output_dir}")
    trainer.train()
    trainer.save_model(training_args.output_dir)


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list



def reset_router_weights(model):
    for name, param in model.named_parameters():
        if "lora_router" in name and "weight" in name:
            # use torch.nn.init.kaiming_uniform_
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))

        if "lora_router" in name and "bias" in name:
            # initialize to zero
            torch.nn.init.zeros_(param)



def _finetune_router(dataset, tokenizer, config, experts, peft_strategy, model_values, path_values, top_k=0):
    output_dir = os.path.join(path_values.output_dir, peft_strategy)

    training_args = _get_training_arguments(config, peft_strategy, f"router_top_k_{top_k}")
    training_args.output_dir = os.path.join(output_dir, f"router_top_k_{top_k}")

    model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
    peft_config.experts_to_combine = list(experts.keys())
    peft_config.num_experts = len(experts.keys())
    peft_config.inference_mode = False
    peft_config.only_router = True
    peft_config.top_k = top_k

    model_peft = get_peft_model(model, peft_config)
    logger.info(f"Loading experts: {experts.keys()}")

    model_peft.load_experts(output_dir, 'default', list(experts.keys()), True)
    model_peft._mark_only_router_as_trainable()
    model_peft.print_trainable_parameters()
    reset_router_weights(model_peft)
    trainer = _init_trainer(model_peft, dataset, peft_config, tokenizer, training_args)

    for name, param in model_peft.named_parameters():
        if param.requires_grad:
            print(name)

    logger.info(f"Peft Config: {peft_config}")
    logger.info(f"Training router, output_dir: {training_args.output_dir}")
    trainer.train()
    trainer.save_model(training_args.output_dir)


def _finetune_experts(dataset, tokenizer, config, experts, peft_strategy, use_base, model_values, path_values):
    output_dir = os.path.join(path_values.output_dir, peft_strategy)

    for expert_name, expert_data in experts.items():
        expert_dataset = dataset.filter(lambda row: str(row["cluster"]) in expert_data["categories"])

        training_args = _get_training_arguments(config, peft_strategy, expert_name)
        training_args.output_dir = os.path.join(output_dir, expert_name)

        if use_base:
            # We want to start from the base adapter weights instead of default initialization weights
            base_path = os.path.join(output_dir, "base")
            model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)

            if peft_config.num_experts != 1:
                peft_config.num_experts = 1
                logger.warning(f"Number of experts is not 1 but we are training the an expert, setting it to 1. num_experts: {peft_config.num_experts}")

            model = get_peft_model(model, peft_config)
            model.load_adapter(base_path, "default")
            model.set_adapter("default")

            # print(model)
            # w0 = model.base_model.model.model.layers[3].self_attn.v_proj.lora_A['default'][0:1]
            # print(w0)
            # wb = load_peft_weights(base_path)
            # w1 = wb['base_model.model.model.layers.3.self_attn.v_proj.lora_A']
            # print(w1)

            # print(torch.allclose(w0, w1))

            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)

            # raise Exception("Base loaded")

            trainer = _init_trainer(model, expert_dataset, None, tokenizer, training_args)
        else:
            base_dataset = dataset.shuffle().select(range(len(dataset) // 10))
            # Append base dataset to expert dataset
            expert_dataset = datasets.concatenate_datasets([base_dataset, expert_dataset])
            model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
            model = get_peft_model(model, peft_config)
            trainer = _init_trainer(model, expert_dataset, peft_config, tokenizer, training_args)

        model.print_trainable_parameters()

        logger.info(f"Training expert: {expert_name}, output_dir: {training_args.output_dir}")
        trainer.train()
        trainer.save_model(training_args.output_dir)


def finetune(model_values, path_values, config, experts, peft_strategy='lora', is_base=False, use_base=False, only_router=False, all=False, top_k=0) -> None:
    dataset = datasets.load_dataset(model_values.dataset, split="train")
    tokenizer = prepare_tokenizer(model_values.model, path_values.cache_dir)

    if is_base:
        _finetune_base(dataset, config, peft_strategy, tokenizer, model_values, path_values)
    elif only_router:
        _finetune_router(dataset, tokenizer, config, experts, peft_strategy, model_values, path_values, top_k)
    elif all:
        _finetune_all(dataset, config, peft_strategy, tokenizer, model_values, path_values)
    else:
        _finetune_experts(dataset, tokenizer, config, experts, peft_strategy, use_base, model_values, path_values)
