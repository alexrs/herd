import os
import datetime

import numpy as np
from peft import LoraModel, PeftConfig
from loguru import logger

class MultiloraModel(LoraModel):
    """
    Based on https://github.com/huggingface/peft/tree/93d0c03d5ba6b2a6b16b7ca887e740a67bc680f3/src/peft/tuners/lora

    Creates a Multi LoRA model from a pretrained transformer model.
    This model uses `add_weighted_adapter` to add adapters to the model with some weights.

    Args:
        model ([`~transformers.PreTrainedModel`]): The model to be adapted.
        adapters (str): The base path to the adapters.
        adapter_names (list): The names of the adapters to be loaded.
        router (herd.Router): The router to be used for routing the adapters.

    Returns:
        `torch.nn.Module`: The Multilora model.
    """

    def __init__(self, model, adapters, adapter_names, router):
        # Build PerftConfig from the first adapter
        config = PeftConfig.from_pretrained(os.path.join(adapters, adapter_names[0]))
        super().__init__(model, config, adapter_names[0])

        # Load the other adapters
        for adapter_name in adapter_names[1:]:
            self.load_adapter(os.path.join(adapters, adapter_name), adapter_name)

        self.router = router

    def generate(self, prompt: str, top: int = 1, **kwargs):
        self.route_to_experts(prompt, top)
        return self.model.generate(**kwargs)

    def route_to_experts(self, instruction: str, top: int = 1):
        # Experts is a list of tuples (expert_name, score).
        experts = self.router.route(instruction, top)
        if top == 1:
            # If we only want the top expert, set it as an adapter
            self.model.set_adapter(experts[0][0])
        else:
            # Otherwise, we compute a new adapter as a combination of the top experts.
            # We generate a unique name for the adapter because even if the same experts are used
            # the weights may be different.
            adapter_name = str(hash(datetime.datetime.now()))

            weights = np.array([expert[1] for expert in experts])
            inverted_weights = 1 / weights
            w = inverted_weights / np.sum(inverted_weights)
            e = [expert[0] for expert in experts]
            logger.debug(f"Creating adapter for: {list(zip(e, w))}")

            # TODO: Experiment with other routing methods
            self.model.add_weighted_adapter(
                e,
                w,
                combination_type="linear",
                adapter_name=adapter_name,
            )

            self.model.set_adapter(adapter_name)

        return experts
