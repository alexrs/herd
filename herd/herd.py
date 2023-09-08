import json
import argparse
import os
from finetune import finetune
from finetune_experts import finetune_experts
from run_model import run_model
from configparser import ConfigParser, ExtendedInterpolation
from dataclasses import dataclass


def main():
    parser = argparse.ArgumentParser(description="Fine-Tune Llama 2 models")
    parser.add_argument(
        "action", choices=["finetune", "finetune_experts", "run_model", "route"]
    )
    parser.add_argument("--config_file", default="config.ini")
    args = parser.parse_args()

    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read(args.config_file)

    models_values = ModelsValues(**dict(config.items("Models")))
    paths_values = PathsValues(**dict(config.items("Paths")))

    # Create base_dir if it does not exists
    if not os.path.exists(paths_values.base_dir):
        os.makedirs(paths_values.base_dir)

    match args.action:
        case "finetune":
            finetune(models_values, paths_values, config)
        case "finetune_experts":
            with open(
                "experts.json", "r"
            ) as json_file:  # TODO: Make the file name configurable
                # Parse the JSON data from the file
                experts = json.loads(json_file.read())
                finetune_experts(models_values, paths_values, config, experts)
        case "run_model":
            with open(
                "experts.json", "r"
            ) as json_file:  # TODO: Make the file name configurable
                # Parse the JSON data from the file
                experts = json.loads(json_file.read())
                run_model(models_values, paths_values, experts)
        case "route":
            raise NotImplementedError("Routing is not implemented yet")


@dataclass
class ModelsValues:
    model: str
    dataset: str
    embeddings_model: str
    embeddings_max_length: int

    def __post_init__(self):
        self.embeddings_max_length = int(self.embeddings_max_length)


@dataclass
class PathsValues:
    base_dir: str
    dataset_dir: str
    cache_dir: str
    output_dir: str


# class HerdModel():
#     def __init__(self, model, expert=None):
#         self.model = model
#         self.expert = expert

#     def __call__(self, input):
#         return self.model(input)

#     def __str__(self):
#         return f"{self.model} {self.expert}"

#     def __repr__(self):
#         return f"{self.model} {self.expert}"


#     def replace_expert(self, new_expert):
#         if self.new_expert == self.expert:
#             return

#         self.expert = new_expert


#     def __eq__(self, other):
#         return self.model == other.model and self.expert == other.expert


#     def __hash__(self):
#         return hash((self.model, self.expert))

#     def __ne__(self, other):
#         return not self.__eq__(other)


if __name__ == "__main__":
    main()
