from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VL_LLM:
    def __init__(self, min_pixels: int = 1, max_pixels: int = 2560, max_new_tokens: int = 512):
        # https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
        self.model_name = "Qwen/Qwen2-VL-7B-Instruct"
        self.max_new_tokens = max_new_tokens
        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            # attn_implementation="flash_attention_2",
            device_map="auto",
        )

        # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
        self.min_pixels = min_pixels*28*28
        self.max_pixels = max_pixels*28*28
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

    def generate_answer(self, context_images: List[Image.Image], query: str) -> torch.Tensor:
        """
        Generate an answer to a user query based on provided context images.

        Args:
            context (List[PIL.Image.Image]): A list of retrieved context images.
            query (str): The user query.

        Returns:
            str: The generated answer as a string.
        """

        messages = [
            {"role": "system", "content": (
                "You are an advanced answer generation assistant in a Retrieval-Augmented Generation (RAG) pipeline. "
                "Your task is to provide accurate, concise, and contextually relevant answers to user queries. "
                "You will receive retrieved images and a specific question to answer. "
                "Use only the provided context to formulate your response, and do not make assumptions beyond the given information. "
                "If the context is insufficient to answer the query, clearly state that more information is needed."
            )},
            {"role": "user", "content": [
                # Create a placeholder for each context image
                *[{"type": "image"} for _ in context_images],
                {"type": "text", "text": query},
            ]}
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=context_images,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        torch.cuda.empty_cache()

        return output_text
