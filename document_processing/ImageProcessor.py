from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

from PIL import Image

from .DescribableAbstract import DescribableAbstract
from .EmbeddableAbstract import EmbeddableAbstract

if TYPE_CHECKING:
    from .DocumentProcessor import DocumentProcessor
    from .FolderProcessor import FolderProcessor
    from .PageProcessor import PageProcessor


class ImageProcessor(EmbeddableAbstract, DescribableAbstract):
    def __init__(self, folder: 'FolderProcessor', document: 'DocumentProcessor', page: 'PageProcessor', index: int, content: Image.Image):
        self.folder: FolderProcessor = folder
        self.document: DocumentProcessor = document
        self.page: PageProcessor = page
        self.index: int = index
        self.content: Image.Image = content
        self.path: Path = Path(f"{page.folder_path}/image_{index}.jpg")
        self.content.save(self.path)

        self.record: Dict[str, Any] = {
            "id": f"{self.document.path.stem}_page_{self.page.number}_image_{self.index}",
            # document is set in the function set_description
            "document": None,
            "metadata": {
                "type": "image",
                "document": self.document.path.name,
                "document_path": str(self.document.path.resolve()),
                "page": self.page.number,
                "page_path": str(self.page.image_path.resolve()),
                "index": self.index,
                "image_path": str(self.path.resolve()),
            },
            # embedding is set in the function set_embedding
            "embedding": None
        }
