from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from typing import List

import torch


class Router:
    def __init__(self, model, tokenizer, experts):
        self.model = model
        self.tokenizer = tokenizer
        self.experts = experts

    def route(self, prompt: str):
        """
        Selects the best expert to answer to prompt
        """
        query_emb = Utils.calculate_embeddings(prompt, self.model, self.tokenizer)

        pass


class Utils:
    @classmethod
    def calculate_embeddings(
        input: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
    ) -> List[float]:
        """
        Calculate an embedding vector for `input`.
        Based on https://github.com/jondurbin/airoboros/blob/237027c46d8b48df7fa2037bcd56711a0587561c/airoboros/embeddings.py#L27
        """
        # Tokenize the input.
        inputs = tokenizer(input, padding=False, truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        # Calculate embeddings for the input.
        embeddings = model.encode(input_ids, normalize_embeddings=True)

        # Calculate the average embedding.
        average_embedding = torch.mean(embeddings, dim=0)

        return average_embedding.tolist()

