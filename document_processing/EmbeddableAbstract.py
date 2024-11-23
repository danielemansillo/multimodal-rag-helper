from torch import Tensor


class EmbeddableAbstract:
    def __init__(self) -> None:
        self.embedding: Tensor = None

    def set_embedding(self, embedding: Tensor):
        self.embedding = embedding
        self.record["embedding"] = self.embedding.tolist()
