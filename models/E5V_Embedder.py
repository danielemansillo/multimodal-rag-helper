from typing import List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


class E5V_Embedder:

    def __init__(self):
        # https://huggingface.co/royokong/e5-v
        self.model_name = 'royokong/e5-v'
        self.llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

        self.text_prompt = self.llama3_template.format(
            '<sent>\nSummary above sentence in one word: ')
        self.image_prompt = self.llama3_template.format(
            '<image>\nSummary above image in one word: ')

        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            'royokong/e5-v', torch_dtype=torch.float16).cuda()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Embed a list of texts in batches.

        Args:
            texts (List[str]): List of text strings to be embedded.
            batch_size (int, optional): Number of texts per batch. Defaults to 32.

        Returns:
            torch.Tensor: Tensor of normalized embeddings for the input texts.
        """
        # Initialize an empty list to store embeddings
        all_embs = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            text_inputs = self.processor([self.text_prompt.replace('<sent>', text) for text in batch_texts],
                                         return_tensors="pt", padding=True).to('cuda')

            with torch.no_grad():
                text_embs = self.model(**text_inputs, output_hidden_states=True,
                                       return_dict=True).hidden_states[-1][:, -1, :]

            text_embs = F.normalize(text_embs, dim=-1)

            # Append the embeddings to the list
            all_embs.extend(text_embs)

            # Clear cache to avoid memory issues (optional, depending on your environment)
            torch.cuda.empty_cache()

        # Concatenate all batch embeddings into one tensor
        return torch.stack(all_embs, dim=0)

    def embed_images(self, images: List[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """
        Embed a list of images in batches.

        Args:
            images (List[PIL.Image.Image]): List of images to be embedded.
            batch_size (int, optional): Number of images per batch. Defaults to 32.

        Returns:
            torch.Tensor: Tensor of normalized embeddings for the input images.
        """
        # Initialize an empty list to store embeddings
        all_embs = []

        # Process images in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]

            img_inputs = self.processor(
                [self.image_prompt] * len(batch_images), batch_images, return_tensors="pt", padding=True).to('cuda')

            with torch.no_grad():
                img_embs = self.model(**img_inputs, output_hidden_states=True,
                                      return_dict=True).hidden_states[-1][:, -1, :]

            img_embs = F.normalize(img_embs, dim=-1)

            # Append the embeddings to the list
            all_embs.extend(img_embs)

            # Clear cache to avoid memory issues (optional, depending on your environment)
            torch.cuda.empty_cache()

        # Concatenate all batch embeddings into one tensor
        return torch.stack(all_embs, dim=0)
