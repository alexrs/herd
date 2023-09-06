# From https://github.com/jondurbin/airoboros/blob/4cf457eaf541d6025a165f27e8596b6a1980bdab/airoboros/embeddings.py
import numpy as np
import torch
from typing import Any, List
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class Embeddings:
    def __init__(
        self, model=None, tokenizer=None, max_length=models_values.embeddings_max_length
    ):
        self.max_length = max_length
        self.model = model if model else Embeddings.get_default_model()
        self.tokenizer = tokenizer if tokenizer else Embeddings.get_default_tokenizer()

    @staticmethod
    def get_default_model(model_name_or_path=models_values.embeddings_model):
        model = SentenceTransformer(
            model_name_or_path, device="cuda", cache_folder=paths_values.cache_dir
        )
        return model

    @staticmethod
    def get_default_tokenizer(model_name_or_path=models_values.embeddings_model):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return tokenizer

    def calculate_fragment_embeddings(self, fragment: str) -> List[float]:
        """Calculate vector embeddings for a single input fragment, which is smaller than the
        max model length.
        """
        with torch.no_grad():
            return self.model.encode(fragment, normalize_embeddings=True)

    def calculate_embeddings(self, input_text: str) -> List[float]:
        """Calculate the vector embeddings for the specified input text.

        1. split the text based on the model's max sequence length
        2. calculate the embeddings for each chunk
        3. calculate the average embedding across all chunks
        """

        # Tokenize the input, and convert tokens into chunks based on max model size.
        inputs = self.tokenizer(
            input_text, padding=False, truncation=False, return_tensors="pt"
        )
        chunks = [
            torch.Tensor(inputs["input_ids"][0][i : i + self.max_length].tolist()).int()
            for i in range(0, len(inputs["input_ids"][0]), self.max_length)
        ]
        fragments = [self.tokenizer.decode(chunk) for chunk in chunks]

        # Now, calculate embeddings for each fragment.
        all_embeddings = []
        lengths = []
        for fragment in fragments:
            lengths.append(len(fragment))
            all_embeddings.append(self.calculate_fragment_embeddings(fragment))

        # Finally, calculate the average across all fragments.
        embeddings = np.average(all_embeddings, axis=0, weights=lengths)
        return embeddings / np.linalg.norm(embeddings)

    # def average_pool(
    #     self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
    # ) -> torch.Tensor:
    #     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    #     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
