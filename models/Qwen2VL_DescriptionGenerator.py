from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VL_DescriptionGenerator:
    def __init__(self, min_pixels: int = 1, max_pixels: int = 1280, max_new_tokens: int = 512):
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

    def generate_image_descriptions(self, page_image: Image.Image, detail_images: List[Image.Image]) -> List[str]:
        """
        Generate detailed descriptions for a list of images using the context of a page image.

        Args:
            page_image (Image.Image): The context image representing the page.
            detail_images (List[Image.Image]): List of images to describe with the page context.

        Returns:
            List[str]: List of detailed descriptions for each image.
        """
        # multi-image conversation, separate images
        descriptions = []

        messages = [
            {"role": "system", "content": (
                "You are a multimodal assistant tasked with generating precise and detailed descriptions of images. "
                "You will describe images by leveraging contextual visual information provided from a related image. "
                "Always focus on the details of the target image while ensuring the description is relevant to the context."
            )},
            {
                "role": "user",
                "content": [
                        {"type": "image"},  # Context image
                        {"type": "image"},  # Detail image
                        {
                            "type": "text",
                            "text": (
                                "The first image shows the entire page containing the second image. "
                                "Analyze the second image in the context of the first one. "
                                "Describe in detail the content of the second image. "
                                "Make your description specific and avoid generalizations. "
                                "Begin your description with phrases like 'The image contains...' or 'The image shows...'."
                            ),
                        },
                ],
            }
        ]

        for image in detail_images:

            images = [page_image, image]

            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                # images=image_inputs,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            descriptions.extend(output_text)

            torch.cuda.empty_cache()

        return descriptions
