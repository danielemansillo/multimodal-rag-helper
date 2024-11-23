from typing import List

import torch
from sentence_transformers import SentenceTransformer


class Stella_Embedder:

    def __init__(self) -> None:
        # https://huggingface.co/dunzhang/stella_en_1.5B_v5
        self.model_name = "dunzhang/stStelella_en_1.5B_v5"
        # This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
        # They are defined in `config_sentence_transformers.json`
        self.query_prompt_name = "s2p_query"

        # ！The default dimension is 1024, if you need other dimensions, please clone the model and modify `modules.json` to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` !
        self.model = SentenceTransformer(
            "dunzhang/stStelella_en_1.5B_v5", trust_remote_code=True).cuda()

    def embed_queries(self, queries: List[str]) -> torch.Tensor:
        return self.model.encode(queries, prompt_name=self.query_prompt_name)

    def embed_documents(self, texts: List[str]) -> torch.Tensor:
        # TODO if necessary implement the batch encoding for the documents
        return self.model.encode(texts)
