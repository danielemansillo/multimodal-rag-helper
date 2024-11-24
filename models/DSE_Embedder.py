from typing import List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def _get_embedding(last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
    reps = last_hidden_state[:, -1]
    reps = torch.nn.functional.normalize(
        reps[:, :dimension], p=2, dim=-1)
    return reps


class DSE_Embedder:
    def __init__(self, min_pixels: int = 1, max_pixels: int = 2560, embedding_dimension: int = 1536):
        # https://huggingface.co/MrLight/dse-qwen2-2b-mrl-v1
        self.model_name = "MrLight/dse-qwen2-2b-mrl-v1"
        self.min_pixels = min_pixels*28*28
        self.max_pixels = max_pixels*28*28
        self.embedding_dimension = embedding_dimension

        self.processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            # TODO Turn it on on the suitable GPUs
            # attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16).to('cuda:0').eval()

        self.processor.tokenizer.padding_side = "left"
        self.model.padding_side = "left"

    def _embed_texts(self, texts: List[str], type: str, batch_size: int = 32) -> torch.Tensor:
        """
        Embed a list of texts in batches.

        Args:
            texts (List[str]): List of text strings to be embedded.
            batch_size (int, optional): Number of texts per batch. Defaults to 32.

        Returns:
            torch.Tensor: Tensor of embeddings for the input texts.
        """
        # Initialize an empty list to store embeddings
        all_embs = []

        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            print("batch_texts")
            print(batch_texts)

            # Prepare the messages for the batch
            doc_messages = [
                [{
                    'role': 'user',
                    'content': [
                        # Adding a dummy image for the easier process
                        {'type': 'image', 'image': Image.new(
                            'RGB', (28, 28)), 'resized_height': 1, 'resized_width': 1},
                        {'type': 'text', 'text': f'{type}: {doc}'}
                    ]
                }]
                for doc in batch_texts
            ]

            # Apply the chat template and add the generation prompt
            doc_texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in doc_messages
            ]

            # Process vision info (dummy image handling)
            doc_image_inputs, doc_video_inputs = process_vision_info(
                doc_messages)

            # Prepare the inputs for the model
            doc_inputs = self.processor(text=doc_texts, images=doc_image_inputs,
                                        videos=doc_video_inputs, padding='longest', return_tensors='pt').to('cuda:0')

            # Prepare the inputs for generation
            cache_position = torch.arange(0, len(doc_texts))
            doc_inputs = self.model.prepare_inputs_for_generation(
                **doc_inputs, cache_position=cache_position, use_cache=False
            )

            # Run the model to get the output
            with torch.no_grad():
                output = self.model(
                    **doc_inputs, return_dict=True, output_hidden_states=True)

            # Extract the embeddings from the hidden states
            doc_embeddings = _get_embedding(
                output.hidden_states[-1], self.embedding_dimension)

            # Append the embeddings of this batch to the list
            all_embs.extend(doc_embeddings)

            # Optionally clear GPU cache after each batch (useful for large datasets)
            torch.cuda.empty_cache()

        # Concatenate all batch embeddings into one tensor
        return torch.stack(all_embs, dim=0)

    def embed_queries(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        return self._embed_texts(texts, "Query", batch_size)

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        return self._embed_texts(texts, "Document", batch_size)

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

            # Prepare messages for the batch with dummy text prompt and images
            doc_messages = [
                [{
                    'role': 'user',
                    'content': [
                        # Adjust image size if needed
                        {'type': 'image', 'image': doc},
                        {'type': 'text', 'text': 'What is shown in this image?'}
                    ]
                }]
                for doc in batch_images
            ]

            # Apply the chat template and prepare text inputs for the batch
            doc_texts = [
                self.processor.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
                for msg in doc_messages
            ]

            # Process vision info (image processing)
            doc_image_inputs, doc_video_inputs = process_vision_info(
                doc_messages)

            # Prepare the inputs for the model
            doc_inputs = self.processor(text=doc_texts, images=doc_image_inputs,
                                        videos=doc_video_inputs, padding='longest', return_tensors='pt').to('cuda:0')

            # Prepare the inputs for generation
            cache_position = torch.arange(0, len(doc_texts))
            doc_inputs = self.model.prepare_inputs_for_generation(
                **doc_inputs, cache_position=cache_position, use_cache=False
            )

            # Run the model to get the output
            with torch.no_grad():
                output = self.model(
                    **doc_inputs, return_dict=True, output_hidden_states=True)

            # Extract the embeddings from the hidden states
            doc_embeddings = _get_embedding(
                output.hidden_states[-1], self.embedding_dimension)

            # Append the embeddings of this batch to the list
            all_embs.extend(doc_embeddings)

            # Optionally clear GPU cache after each batch (useful for large datasets)
            torch.cuda.empty_cache()

        # Concatenate all batch embeddings into one tensor
        return torch.stack(all_embs, dim=0)
