from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import pymupdf
from PIL import Image

from .DescribableAbstract import DescribableAbstract
from .EmbeddableAbstract import EmbeddableAbstract

if TYPE_CHECKING:
    from .DocumentProcessor import DocumentProcessor
    from .FolderProcessor import FolderProcessor

from .utils import (apply_mask, paste_on_white_background, pixmap_to_image,
                    resize_image_to_target_area, validate_length)


class PageProcessor(EmbeddableAbstract, DescribableAbstract):
    def __init__(self, folder: 'FolderProcessor', document: 'DocumentProcessor', number: int, page: pymupdf.Page):
        from .ImageProcessor import ImageProcessor
        from .TextProcessor import TextProcessor

        self.folder: FolderProcessor = folder
        self.document: DocumentProcessor = document
        self.number: int = number
        self.page: pymupdf.Page = page
        self.image_path: Path = Path(
            f"{document.path.with_suffix('')}/page_{number}.jpg")

        # Initialize the main page image
        self.image: Image.Image = self._process_page_image(page)
        self.image.save(fp=self.image_path)

        # Extract and initialize text content
        self.text: TextProcessor = TextProcessor(
            folder, document, self, content=self.page.get_text())

        document_images = self._extract_page_images(document, page)
        if len(document_images) > 0:
            self.folder_path: Path = Path(
                f"{document.path.with_suffix('')}/page_{number}")
            self.folder_path.mkdir(parents=True, exist_ok=True)

        # Extract and initialize individual images on the page
        self.images: List[ImageProcessor] = [ImageProcessor(
            folder, document, self, index, content=image) for index, image in enumerate(document_images)]

        self.record: Dict[str, Any] = {
            "id": f"{self.document.path.stem}_page_{self.page.number}",
            # document is set in the function set_description
            "document": None,
            "metadata": {
                "type": "page",
                "document": self.document.path.name,
                "document_path": str(self.document.path.resolve()),
                "page": self.page.number,
                "page_path": str(self.image_path.resolve())
            },
            # embedding is set in the function set_embedding
            "embedding": None
        }

    def set_image_descriptions(self, descriptions: List[str]) -> None:
        """
        Set the description for the images in the page and for the page itself by concatenating the text and the images descriptions

        Args:
            descriptions (List[str]): _description_
        """
        validate_length(self.images, descriptions, "images", "descriptions")
        for image, description in zip(self.images, descriptions):
            image.set_description(description=description)
        
        self.set_description("Text: " + self.text.content + "\n".join([f"Image{i}: {image.description}" for i, image in enumerate(self.images, start=1)]))

    def _process_page_image(self, page: pymupdf.Page) -> Image.Image:
        """
        Processes the primary page image by rendering it and resizing it.

        Args:
            page (pymupdf.Page): The page to render.

        Returns:
            Image: The processed and resized page image.
        """
        pixmap = page.get_pixmap(dpi=200)
        image = pixmap_to_image(pixmap)
        resized_image = resize_image_to_target_area(image)
        return resized_image

    def _extract_page_images(self,
                             document: 'DocumentProcessor',
                             page: pymupdf.Page) -> List['ImageProcessor']:
        """
        Extracts and processes individual images on the page.

        Args:
            folder (FolderProcessor): The folder containing the document.
            document (DocumentProcessor): The parent document processor.
            page (pymupdf.Page): The page to extract images from.

        Returns:
            List[ImageProcessor]: A list of processed images from the page.
        """
        images = []
        for pixmap in page.get_images(full=True):
            image = self._process_individual_image(
                document.document, pixmap)
            images.append(image)
        return images

    def _process_individual_image(self,
                                  document: pymupdf.Document,
                                  pixmap: pymupdf.Pixmap) -> Image:
        """
        Processes an individual image, applying any necessary transformations.

        Args:
            document (pymupdf.Document): The parent document.
            raw_img (Any): The raw image data extracted from the page.

        Returns:
            Image: The processed image.
        """
        pixmap_w_mask = apply_mask(document, pixmap)
        image = pixmap_to_image(pixmap_w_mask)

        if image.mode == "RGBA":
            image = paste_on_white_background(image)

        return image
