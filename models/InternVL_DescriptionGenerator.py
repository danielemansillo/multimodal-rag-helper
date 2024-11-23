from typing import List

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class InternVL_DescriptionGenerator:
    def __init__(self, max_new_tokens: int = 1024):
        # If you want to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
        # https://huggingface.co/OpenGVLab/InternVL2-8B
        self.model_name = 'OpenGVLab/InternVL2-8B'
        self.max_new_tokens = max_new_tokens
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            # TODO check if flass_attention works on any kind of system
            use_flash_attn=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False)

        # set the max number of tiles in `max_num`
        self.generation_config = dict(
            max_new_tokens=self.max_new_tokens, do_sample=True)

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
        context_image = load_image(
            page_image, max_num=12).to(torch.bfloat16).cuda()

        for image in detail_images:

            detail_image = load_image(
                image.content, max_num=12).to(torch.bfloat16).cuda()

            pixel_values = torch.cat((context_image, detail_image), dim=0)
            num_patches_list = [
                context_image.size(0), detail_image.size(0)]

            question = (
                "<image>",
                "The first image shows the entire page containing the second image. "
                "Analyze the second image in the context of the first one. "
                "Describe in detail the content of the second image. "
                "Make your description specific and avoid generalizations. "
                "Begin your description with phrases like 'The image contains...' or 'The image shows...'."
            )
            response = self.model.chat(self.tokenizer, pixel_values, question, self.generation_config,
                                       num_patches_list=num_patches_list,
                                       history=None, return_history=False)

            descriptions.append(response)

        return descriptions


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
