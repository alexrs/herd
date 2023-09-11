from dataclasses import dataclass


@dataclass
class ModelValues:
    model: str
    dataset: str
    embeddings_model: str
    embeddings_max_length: int

    def __post_init__(self):
        self.embeddings_max_length = int(self.embeddings_max_length)


@dataclass
class PathValues:
    base_dir: str
    dataset_dir: str
    cache_dir: str
    output_dir: str
    experts_dir: str
    experts_file: str
