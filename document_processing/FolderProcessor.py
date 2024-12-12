from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm


class FolderProcessor:
    """Handles processing of a folder containing multiple documents."""

    def __init__(self, path: Path, limit_documents: int = None, pdf_only: bool = True):
        from .DocumentProcessor import DocumentProcessor
        from .ImageProcessor import ImageProcessor
        from .PageProcessor import PageProcessor
        from .TextProcessor import TextProcessor
        self.path: Path = path
        if pdf_only:
            files = list(self.path.glob("*.pdf"))
        else:
            files = list(self.path.iterdir())
            
        self.documents: List[DocumentProcessor] = [
            DocumentProcessor(self, file_path)
            for file_path in tqdm(files, desc="Processing documents")
            if file_path.is_file()
        ]
        if limit_documents:
            self.documents = self.documents[:limit_documents]

        self.all_pages: List[PageProcessor] = []
        self.all_texts: List[TextProcessor] = []
        self.all_images: List[ImageProcessor] = []

        for document in self.documents:
            self.all_pages.extend(document.pages)
            self.all_texts.extend(document.all_texts)
            self.all_images.extend(document.all_images)

    def get_embedding_records(self) -> Dict[str, Dict[str, List[Any]]]:
        # Initialize the combined records with the same structure as individual records
        records = {
            "text": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
            "image": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
            "page": {"ids": [], "documents": [], "metadatas": [], "embeddings": []},
        }

        # Aggregate records from all DocumentProcessors
        for document in self.documents:
            document_records = document.get_embedding_records()

            # Merge records for text, image, and page
            for key in ["text", "image", "page"]:
                for record_type in ["ids", "documents", "metadatas", "embeddings"]:
                    records[key][record_type].extend(
                        document_records[key][record_type])

        return records
