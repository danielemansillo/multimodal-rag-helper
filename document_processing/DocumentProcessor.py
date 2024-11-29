import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import pymupdf
from torch import Tensor

from .utils import validate_length

if TYPE_CHECKING:
    from .FolderProcessor import FolderProcessor


class DocumentProcessor:
    def __init__(self, folder: 'FolderProcessor', path: Path):
        from .ImageProcessor import ImageProcessor
        from .PageProcessor import PageProcessor
        from .TextProcessor import TextProcessor

        self.folder: FolderProcessor = folder
        self.path: Path = path
        # Create a folder to contain the page images and the images extracted
        self.folder_path: Path = self.path.with_suffix("")
        self.folder_path.mkdir(parents=True, exist_ok=True)

        self.document: pymupdf.Document = pymupdf.open(path)

        self.pages = [PageProcessor(folder, self, index, page)
                      for index, page in enumerate(self.document)]

        self.all_texts: List[TextProcessor] = [
            page.text for page in self.pages]

        self.all_images: List[ImageProcessor] = []
        for page in self.pages:
            self.all_images.extend(page.images)

        self.descriptions_path: Path = self.path.with_suffix(".json")
        if self.descriptions_path.exists():
            self.load_descriptions()

    def set_image_page_descriptions(self, descriptions: List[str]) -> None:
        """
        Set the descriptions for all the images in the document and then set the descriptions for the pages by concatenating the text and the images descriptions.

        Args:
            descriptions (List[str]): List of all the images descriptions in the document
        """
        validate_length(
            self.all_images, descriptions, "images", "descriptions")

        for image, description in zip(self.all_images, descriptions):
            image.set_description(description=description)

        for page in self.pages:
            page_description = "Text: " + page.text.content + \
                "\n".join([f"Image{i}: {image.description}" for i,
                          image in enumerate(page.images, start=1)])
            page.set_description(page_description)

    def set_page_descriptions(self, descriptions: List[str]) -> None:
        validate_length(
            self.pages, descriptions, "pages", "descriptions")

        for page, description in zip(self.pages, descriptions):
            page.set_description(description)

    def set_text_embeddings(self, embeddings: Tensor) -> None:
        """Set embeddings for text elements in pages."""
        validate_length(self.pages, embeddings, "texts", "embeddings")
        for page, embedding in zip(self.pages, embeddings):
            page.text.set_embedding(embedding=embedding)

    def set_image_embeddings(self, embeddings: Tensor) -> None:
        """Set embeddings for images."""
        validate_length(self.all_images, embeddings, "images", "embeddings")
        for image, embedding in zip(self.all_images, embeddings):
            image.set_embedding(embedding=embedding)

    def set_page_embeddings(self, embeddings: Tensor) -> None:
        """Set embeddings for pages."""
        validate_length(self.pages, embeddings, "pages", "embeddings")
        for page, embedding in zip(self.pages, embeddings):
            page.set_embedding(embedding=embedding)

    def get_embedding_records(self) -> Dict[str, Dict[str, List[Any]]]:
        records = {
            "text": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
            "image": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
            "page": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
        }
        record_keys_mapping = {"ids": "id", "documents": "document",
                               "metadatas": "metadata", "embeddings": "embedding"}

        # Populate records
        for page in self.pages:
            for records_key, record_key in record_keys_mapping.items():
                # Page-level records
                records["page"][records_key].append(page.record[record_key])

                # Text-level records
                records["text"][records_key].append(
                    page.text.record[record_key])

            # Image-level records
            for image in page.images:
                for records_key, record_key in record_keys_mapping.items():
                    records["image"][records_key].append(
                        image.record[record_key])

        return records

    def dump_descriptions(self):
        images_descriptions = {}
        for i, page in enumerate(self.pages):
            images_descriptions[f"Page {i}"] = {}
            for j, image in enumerate(page.images):
                images_descriptions[f"Page {i}"][f"Image {j}"] = image.description

        with self.descriptions_path.open("w", encoding="utf-8") as json_file:
            json.dump(images_descriptions, json_file,
                      indent=2, ensure_ascii=False)

    def load_descriptions(self):
        descriptions_list = []
        with self.descriptions_path.open("r", encoding="utf-8") as json_file:
            descriptions_dict = json.load(json_file)
        for i, page in enumerate(self.pages):
            for j, image in enumerate(page.images):
                descriptions_list.append(
                    descriptions_dict[f"Page {i}"][f"Image {j}"])
        self.set_image_page_descriptions(descriptions=descriptions_list)
