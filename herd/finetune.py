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
        run_name=os.path.join(peft_strategy, expert_name),
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
    dataset = dataset.shuffle()

    model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
    model_to_train = get_peft_model(model, peft_config)
    model_to_train.print_trainable_parameters()

    for name, param in model_to_train.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    expert_name = f"all_r_{peft_config.r}"
    if peft_config.peft_type == "MOLORA" and peft_config.self_attn_router:
        expert_name += "_self_attn_" + str(peft_config.self_attn_hidden_dim)
        if peft_config.self_attn_use_value:
            expert_name += "_use_value"

    training_args = _get_training_arguments(config, peft_strategy, expert_name)
    output_dir = os.path.join(path_values.output_dir, peft_strategy, expert_name)
    training_args.output_dir = output_dir

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

def _finetune_router(
        dataset,
        tokenizer,
        config,
        experts,
        peft_strategy,
        model_values,
        path_values,
    ):

    # get unique values of original_dataset
    original_datasets = dataset.unique("cluster")

    # # get the count of hendrycks/competition_math (smallest dataset)
    # min_original_dataset_count = len(dataset.filter(lambda x: x["cluster"] == 5))
    # print(min_original_dataset_count)
    # raise Exception("stop")
    min_original_dataset_count = 460

    # # get samples for each dataset
    dataset_samples = {}
    for original_dataset in original_datasets:
        dataset_samples[original_dataset] = dataset.filter(lambda x: x["cluster"] == original_dataset).select(range(min_original_dataset_count))

    # # concatenate all the samples
    dataset = datasets.concatenate_datasets(list(dataset_samples.values()))

    # shuffle the dataset
    dataset = dataset.shuffle()

    print(f"LEN DATASET: {len(dataset)}")

    output_dir = os.path.join(path_values.output_dir, peft_strategy)

    model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
    peft_config.experts_to_combine = list(experts.keys())
    peft_config.num_experts = len(experts.keys())
    peft_config.inference_mode = False
    peft_config.only_router = True
    # peft_config.self_attn_router = model_values.self_attn_router
    # peft_config.self_attn_hidden_dim = model_values.self_attn_hidden_dim
    # peft_config.self_attn_use_value = model_values.self_attn_use_value


    model_name = "router"

    if peft_config.self_attn_router:
        model_name += "_self_attn_" + str(peft_config.self_attn_hidden_dim) + "_norm"
        if peft_config.self_attn_use_value:
            model_name += "_use_value"

    if peft_config.router_dropout > 0:
        model_name += "_dropout_" + str(peft_config.router_dropout)

    training_args = _get_training_arguments(config, peft_strategy, model_name)
    training_args.output_dir = os.path.join(output_dir, model_name)

    print(peft_config)

    model_peft = get_peft_model(model, peft_config)

    print(model_peft)

    logger.info(f"Loading experts: {experts.keys()}")

    model_peft.load_experts(output_dir, 'default', list(experts.keys()), True)
    model_peft._mark_only_router_as_trainable()
    model_peft.print_trainable_parameters()
    reset_router_weights(model_peft)
    trainer = _init_trainer(model_peft, dataset, peft_config, tokenizer, training_args)

    for name, param in model_peft.named_parameters():
        if param.requires_grad:
            print(name, param.requires_grad)

    logger.info(f"Peft Config: {peft_config}")
    logger.info(f"Training router, output_dir: {training_args.output_dir}")
    trainer.train()
    trainer.save_model(training_args.output_dir)


def _finetune_experts(dataset, tokenizer, config, experts, peft_strategy, use_base, model_values, path_values):
    output_dir = os.path.join(path_values.output_dir, peft_strategy)

    for expert_name, expert_data in experts.items():
        expert_dataset = dataset.filter(lambda row: str(row["cluster"]) in expert_data["categories"])

        print(f"Training on {len(expert_dataset)} samples")

        training_args = _get_training_arguments(config, peft_strategy, expert_name)
        training_args.output_dir = os.path.join(output_dir, expert_name)

        if use_base:
            # We want to start from the base adapter weights instead of default initialization weights
            base_path = os.path.join(output_dir, "base")
            model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)

            if peft_config.num_experts != 1:
                peft_config.num_experts = 1
                logger.warning(f"Number of experts is not 1 but we are training an expert, setting it to 1. num_experts: {peft_config.num_experts}")

            model = get_peft_model(model, peft_config)
            model.load_adapter(base_path, "default")
            model.set_adapter("default")

            trainer = _init_trainer(model, expert_dataset, None, tokenizer, training_args)
        else:
            # Append base dataset to expert dataset
            model, peft_config = prepare_model(model_values.model, path_values.cache_dir, config, peft_strategy)
            if peft_config.num_experts != 1:
                peft_config.num_experts = 1
                logger.warning(f"Number of experts is not 1 but we are training the base expert, setting it to 1. num_experts: {peft_config.num_experts}")

            model = get_peft_model(model, peft_config)
            trainer = _init_trainer(model, expert_dataset, peft_config, tokenizer, training_args)

        print(model)
        model.print_trainable_parameters()

        logger.info(f"Training expert: {expert_name}, output_dir: {training_args.output_dir}")
        trainer.train()
        trainer.save_model(training_args.output_dir)



def finetune(
        model_values,
        path_values,
        config,
        experts,
        peft_strategy='lora',
        is_base=False,
        use_base=False,
        only_router=False,
        all=False,
    ) -> None:
    dataset = datasets.load_dataset(model_values.dataset, split="train")

    # # get only first 25% of the data for original_dataset = Open-Orca/OpenOrca
    # # Separate the "Open-Orca/OpenOrca" data
    # open_orca_data = dataset.filter(lambda x: x['original_dataset'] == 'Open-Orca/OpenOrca')
    # # Calculate the midpoint to slice the "Open-Orca/OpenOrca" data in half
    # midpoint = len(open_orca_data) // 6
    # # Keep only the first half of the "Open-Orca/OpenOrca" rows
    # part_open_orca = open_orca_data.select(range(midpoint))

    # code_alpaca_data = dataset.filter(lambda x: x['original_dataset'] == 'TokenBender/code_instructions_122k_alpaca_style')
    # part_code_alpaca = code_alpaca_data.select(range(len(code_alpaca_data) // 2))

    # alpaca_cleaned_data = dataset.filter(lambda x: x['original_dataset'] == 'yahma/alpaca-cleaned')
    # part_alpaca_cleaned = alpaca_cleaned_data.select(range(len(alpaca_cleaned_data) // 2))

    # # Separate the remaining data
    # other_data = dataset.filter(lambda x: x['original_dataset'] not in ['Open-Orca/OpenOrca', 'TokenBender/code_instructions_122k_alpaca_style', 'yahma/alpaca-cleaned'])

    # # Concatenate the data
    # dataset = datasets.concatenate_datasets([part_open_orca, part_code_alpaca, part_alpaca_cleaned, other_data])

    # # filter out from qwedsacf/grade-school-math-instructions
    # dataset = dataset.filter(lambda x: x['original_dataset'] != 'qwedsacf/grade-school-math-instructions')

    tokenizer = prepare_tokenizer(model_values.model, path_values.cache_dir)

    if is_base:
        _finetune_base(dataset, config, peft_strategy, tokenizer, model_values, path_values)
    elif only_router:
        _finetune_router(
            dataset,
            tokenizer,
            config,
            experts,
            peft_strategy,
            model_values,
            path_values,
        )
    elif all:
        _finetune_all(dataset, config, peft_strategy, tokenizer, model_values, path_values)
    else:
        _finetune_experts(dataset, tokenizer, config, experts, peft_strategy, use_base, model_values, path_values)
