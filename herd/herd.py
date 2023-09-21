import argparse
import json
import os
from configparser import ConfigParser, ExtendedInterpolation

from finetune import finetune
from models import ModelValues, PathValues
from run_model import run_model
from segment_experts import segment_experts


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Llama 2 models")
    parser.add_argument(
        "action",
        choices=["finetune", "run_model", "segment_experts"],
    )
    parser.add_argument("--config-file", default="config.ini")
    parser.add_argument("--only-base", default=False, type=bool)
    parser.add_argument("--interactive", default=False, type=bool)
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
        # Process based on action
        match args.action:
            case "finetune":
                finetune(model_values, path_values, config, experts)
            case "run_model":
                run_model(model_values, path_values, experts, args.only_base, args.interactive)
            case "segment_experts":
                segment_experts(model_values, path_values, experts)


if __name__ == "__main__":
    main()
