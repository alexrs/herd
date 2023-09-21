"""
A dedicated helper to manage templates and prompt building.

From: https://github.com/arielnlee/Platypus/blob/dc90c1a7acf2930f0c89f6b3c9cd24f99efb0295/utils/prompter.py

# TODO: Does this work with SFTTrainer's formatting_func?
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        sample,
        use_output: bool = True,
    ) -> str:
        if sample["input"]:
            res = self.template["prompt_input"].format(
                instruction=sample["instruction"], input=sample["input"]
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=sample["instruction"]
            )
        if use_output and sample["output"]:
            res = f"{res}{sample['output']}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()