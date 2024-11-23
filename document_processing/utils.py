import math

import pymupdf
from PIL import Image

# This is the max size accepted by Claude and it seemed a good compromise


def pixmap_to_image(pix: pymupdf.Pixmap) -> Image.Image:
    # Convert Pixmap to PIL Image
    mode = "RGBA" if pix.alpha else "RGB"
    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    return image


def apply_mask(doc: pymupdf.Document, img: list) -> pymupdf.Pixmap:
    xref = img[0]  # get the XREF of the image
    smask = img[1]  # get the MASK of the image

    pix = pymupdf.Pixmap(doc, xref)  # create a Pixmap for the image

    # If there is a mask apply it (Usually transparency)
    if smask != 0:
        # create a Pixmap for the mask
        mask = pymupdf.Pixmap(doc, smask)
        # Remove alpha channel if present
        if pix.alpha == 1:
            pix = pymupdf.Pixmap(pix, 0)

        # Apply the mask to the image
        pix = pymupdf.Pixmap(pix, mask)

    return pix


def resize_image_to_target_area(img: Image.Image, target_area: int = 1192464) -> Image.Image:
    original_width, original_height = img.size
    # original_area = original_width * original_height

    # # Check if resizing is necessary
    # if original_area <= target_area:
    #     return img  # Return original image if it's already smaller than the target area

    # Calculate aspect ratio and new dimensions
    aspect_ratio = original_width / original_height
    new_height = int(math.sqrt(target_area / aspect_ratio))
    new_width = int(target_area / new_height)

    # Resize the image
    img.thumbnail((new_width, new_height))

    return img


def paste_on_white_background(image: Image.Image) -> Image.Image:
    # Create a white background image of the same size
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))

    # Paste the original image on the white background and convert to RGB
    combined = Image.alpha_composite(background, image).convert("RGB")

    return combined
