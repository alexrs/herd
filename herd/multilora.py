import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from peft import LoraModel, PeftConfig
from peft.tuners.lora import LoraLayer, Linear, Linear4bit, Linear8bitLt, Embedding

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

    # TODO: add support for add_weighted_adapter
    def __init__(self, model, adapters, adapter_names, router):
        # Build PerftConfig from the first adapter
        config = PeftConfig.from_pretrained(os.path.join(adapters, adapter_names[0]))

        super().__init__(model, config, adapter_names[0])

        # Load the other adapters
        for adapter_name in adapter_names[1:]:
            self.load_adapter(os.path.join(adapters, adapter_name), adapter_name)

        # Do we need the router here?
        self.router = router

    def generate(self, prompt, **kwargs):
        experts = self.router.route(prompt)
        self.set_adapter(experts[0][0])
        return self.model.generate(**kwargs)



# class MultiloraLayer(LoraLayer):
#     pass


# class MultiloraLinear(Linear):
#     pass


# class MultiloraLinear4bit(Linear4bit):
#     pass


# class MultiloraLinear8bitLt(Linear8bitLt):
#     pass


# class MultiloraEmbedding(Embedding):
#     pass

