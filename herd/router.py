from typing import Any

import torch.nn.functional as F
from .embeddings import Embeddings
import math
from loguru import logger
import random
import faiss
import numpy as np
import json
from tqdm import tqdm


class Router:
    def __init__(self, model, tokenizer, embeddings: Embeddings, experts, k: int = 50):
        """
        Initializes the router

        Params:
            model: The model to use for calculating embeddings
            tokenizer: The tokenizer to use for tokenizing input
            experts: A list of experts to route to
            k: The number of nearest neighbors to consider when routing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.experts = experts
        self.embeddings = embeddings
        self.k = k

    def route(self, prompt: str):
        """
        Selects the best expert to answer to prompt
        """
        query_emb = self.embeddings.calculate_embeddings(
            prompt, self.model, self.tokenizer
        )
        best_expert = None
        best_distance = math.inf
        for expert, index in self.indices.items():
            distances, _ = index.search(query_emb, k=min(index.ntotal, self.k))
            distances = distances[0].tolist()
            average_distance = sum(distances) / len(distances)
            logger.debug(f"Average distance [{expert}]: {average_distance}")
            if average_distance < best_distance:
                best_distance = average_distance
                best_expert = expert
        logger.success(f"Routing to {best_expert} with score: {best_distance}")
        return best_expert

    def create_index(self, input_path: str) -> Any:
        """Create a faiss index from the routing data for a given expert."""
        logger.info(f"Creating routing faiss index: {input_path}")
        index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        all_items = []
        with open(input_path, "r") as infile:
            for line in infile.readlines():
                all_items.append(json.loads(line)["instruction"])
        random.shuffle(all_items)
        for item in tqdm(all_items[0 : self.max_samples]):
            index.add(
                np.array(
                    [
                        self.embeddings.calculate_embeddings(
                            item, self.model, self.tokenizer
                        )
                    ]
                )
            )
        return index
