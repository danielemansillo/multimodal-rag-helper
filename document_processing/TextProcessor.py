from typing import TYPE_CHECKING, Any, Dict

from .EmbeddableAbstract import EmbeddableAbstract

if TYPE_CHECKING:
    from .DocumentProcessor import DocumentProcessor
    from .FolderProcessor import FolderProcessor
    from .PageProcessor import PageProcessor


class TextProcessor(EmbeddableAbstract):
    def __init__(self, folder: 'FolderProcessor', document: 'DocumentProcessor', page: 'PageProcessor', content: str):
        self.folder: FolderProcessor = folder
        self.document: DocumentProcessor = document
        self.page: PageProcessor = page
        self.content: str = content

        self.record: Dict[str, Any] = {
            "id": f"{self.document.path.stem}_page_{self.page.number}_text",
            "document": self.content,
            "metadata": {
                "type": "text",
                "document": self.document.path.name,
                "document_path": self.document.path.resolve(),
                "page": self.page.number,
                "page_path": self.page.image_path.resolve(),
            },
            # embedding is set in the function set_embedding
            "embedding": None
        }
