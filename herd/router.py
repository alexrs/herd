from typing import Any

from herd.embeddings import Embeddings
from loguru import logger
import faiss
import numpy as np
import os
from typing import List, Tuple


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

    def route(self, prompt: str, top: int = 1) -> List[Tuple[str, float]]:
        """
        Selects the best expert to answer to prompt
        """
        if top > len(self.indices):
            logger.warning(
                f"Requested more experts than available. Setting top to {len(self.indices)}"
            )
            top = len(self.indices)

        query_emb = self.embeddings.calculate_embeddings(prompt)
        query_emb = query_emb.reshape(1, query_emb.shape[0])
        expert_distances = []
        for expert, index in self.indices.items():
            distances, _ = index.search(query_emb, k=min(index.ntotal, self.k))
            distances = distances[0].tolist()
            average_distance = sum(distances) / len(distances)
            logger.debug(f"Average distance [{expert}]: {average_distance}")
            expert_distances.append((expert, average_distance))
        sorted_experts = sorted(expert_distances, key=lambda x: x[1])
        logger.success(f"Routing to {sorted_experts[:top]}")
        return sorted_experts[:top]

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
