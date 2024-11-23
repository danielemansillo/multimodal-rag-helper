from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Qwen2_5_LLM:
    def __init__(self, max_new_tokens: int = 512):
        # https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.max_new_tokens = max_new_tokens

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def generate_answer(self, context: List[str], query: str) -> torch.Tensor:
        """
        Generate an answer to a user query based on provided context.

        Args:
            context (List[str]): A list of retrieved context documents or text segments.
            query (str): The user query.

        Returns:
            str: The generated answer as a string.
        """
        # Combine the context into a single block for the prompt
        context_text = "\n".join(context)

        prompt = ("Context:"
                  f"{context_text}"
                  "Query:"
                  f"{query}")

        messages = [
            {"role": "system", "content": (
                "You are an advanced answer generation assistant in a Retrieval-Augmented Generation (RAG) pipeline. "
                "Your task is to provide accurate, concise, and contextually relevant answers to user queries. "
                "You will receive retrieved documents or contextual information and a specific question to answer. "
                "Use only the provided context to formulate your response, and do not make assumptions beyond the given information. "
                "If the context is insufficient to answer the query, clearly state that more information is needed."
            )},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer(
            [text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True)[0]

        torch.cuda.empty_cache()

        return response
