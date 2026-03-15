from abc import ABC, abstractmethod
from typing import Dict, Any, List

class IConfluenceClient(ABC):
    """Interface for Confluence API communications."""
    
    @abstractmethod
    def get_page(self, page_id: str) -> Dict[str, Any]:
        """Fetches a page by ID and returns its raw data."""
        pass
        
    @abstractmethod
    def get_child_pages(self, page_id: str) -> List[Dict[str, Any]]:
        """Fetches the children of a given page."""
        pass

class IContentParser(ABC):
    """Interface for parsing raw Confluence content into desired format."""
    
    @abstractmethod
    def parse(self, raw_content: str) -> str:
        """Parses the raw HTML content into a cleaner format (e.g., Markdown)."""
        pass

class IStorageBackend(ABC):
    """Interface for persisting extracted data."""
    
    @abstractmethod
    def save(self, content: str, filename: str) -> None:
        """Saves the content to a storage backend."""
        pass
