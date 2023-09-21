import argparse
import json
import os
from configparser import ConfigParser, ExtendedInterpolation

from herd.finetune import finetune
from herd.models import ModelValues, PathValues
from herd.run_model import run_model
from herd.segment_experts import segment_experts
from herd.finetune_2 import finetune2
from herd.test_molora import test_molora

def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Llama 2 models")
    parser.add_argument(
        "action",
        choices=["finetune", "run_model", "segment_experts", "finetune2", "test_molora"],
    )
    parser.add_argument("--config-file", default="config/config.ini")
    parser.add_argument("--only-base", default=False, type=bool)
    parser.add_argument("--interactive", default=False, type=bool)
    parser.add_argument("--peft-strategy", default="lora", type=str)
    parser.add_argument("--top", default=1, type=int)
    parser.add_argument("--is-base", default=False, type=bool)
    parser.add_argument("--use-base", default=False, type=bool)
    parser.add_argument("--expert", default=None, type=str)
    parser.add_argument("--experts-to-train", default=None, type=str)
    parser.add_argument("--only-router", default=False, type=bool)
    parser.add_argument("--all", default=False, type=bool)
    parser.add_argument("--top-k", default=0, type=int)
    args = parser.parse_args()

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config_file)

    model_values = ModelValues(**dict(config.items("Models")))
    path_values = PathValues(**dict(config.items("Paths")))

    # Create base_dir if it does not exists
    if not os.path.exists(path_values.base_dir):
        os.makedirs(path_values.base_dir)

    # Read experts.json file
    with open(path_values.experts_file, "r") as json_file:
        experts = json.loads(json_file.read())
        experts_to_train = args.experts_to_train.split(",") if args.experts_to_train is not None else None
        # Filter experts to train
        if experts_to_train is not None:
            experts = {k: v for k, v in experts.items() if k in experts_to_train}

        # Process based on action
        match args.action:
            case "finetune":
                finetune(model_values, path_values, config, experts, args.peft_strategy, args.is_base, args.use_base, args.only_router, args.all, args.top_k)
            case "run_model":
                run_model(
                    model_values, path_values, experts, args.only_base, args.interactive, args.peft_strategy, args.top
                )
            case "segment_experts":
                segment_experts(model_values, path_values, experts)
            case "finetune2":
                finetune2(model_values, path_values, config, experts, args.peft_strategy, args.is_base, args.expert)
            case "test_molora":
                test_molora(model_values, path_values, config, experts)


if __name__ == "__main__":
    main()
