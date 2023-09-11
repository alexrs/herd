from typing import Any

from herd.embeddings import Embeddings
import math
from loguru import logger
import faiss
import numpy as np
import os


class Router:
    def __init__(
        self,
        embeddings: Embeddings,
        experts,
        k: int = 50,
    ):
        """
        Initializes the router
        """
        self.experts = experts
        self.embeddings = embeddings
        self.k = k

        self.indices = {}
        for expert_name, _ in experts.items():
            # TODO: Should we make the experts folder configurable?
            expert_path = os.path.join("experts", expert_name + ".npy")
            self.indices[expert_name] = self.create_index(expert_path)

    def route(self, prompt: str):
        """
        Selects the best expert to answer to prompt
        """
        query_emb = self.embeddings.calculate_embeddings(prompt)
        query_emb = query_emb.reshape(1, query_emb.shape[0])
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
        index = faiss.IndexFlatL2(
            self.embeddings.model.get_sentence_embedding_dimension()
        )
        em = np.load(input_path)
        em = em.reshape(1, em.shape[0])
        index.add(em)

        return index
