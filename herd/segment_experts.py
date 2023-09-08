import numpy as np
import json
import datasets
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from embeddings import Embeddings
from loguru import logger


# TODO: Read model from config
def main():
    with open("experts.json", "r") as experts_file:
        experts = json.load(experts_file)
        dataset = datasets.load_dataset("jondurbin/airoboros-2.1", split="train")
        model = SentenceTransformer("thenlper/gte-small", device="cuda")
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        max_length = 512

        embeddings = Embeddings(model, tokenizer, max_length)

        for expert_name, expert_data in experts.items():
            logger.info(f"Calculating embedding for {expert_name}...")

            expert_dataset = dataset.filter(
                lambda row: row["category"] in expert_data["categories"]
            )
            # get 100 random samples from the expert dataset. TODO: Is 100 a good number?
            expert_dataset = expert_dataset.select(
                np.random.choice(expert_dataset.shape[0], 100)
            )

            # create an empty numpy array to store embeddings
            es = np.empty((0, embeddings.model.get_sentence_embedding_dimension()))
            # compute embeddings for each sample
            for sample in expert_dataset:
                e = embeddings.calculate_embeddings(sample["instruction"])
                es = np.vstack((es, e))

            # compute average embedding. TODO: Is average a good choice? What about max_pooling?
            avg_embedding = np.average(es, axis=0)

            # save embedding to file
            np.save(f"experts/{expert_name}.npy", avg_embedding)
            logger.info(
                f"  Saved embedding for {expert_name} at experts/{expert_name}.npy."
            )


if __name__ == "__main__":
    main()
