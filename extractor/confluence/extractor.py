import logging
from base import BaseExtractor
from .interfaces import IConfluenceClient, IContentParser, IStorageBackend

logger = logging.getLogger(__name__)

class ConfluencePageExtractor(BaseExtractor):
    """
    Orchestrator class for Confluence.
    Follows Dependency Inversion by depending on abstract interfaces
    instead of concrete implementations.
    """
    
    def __init__(
        self, 
        client: IConfluenceClient, 
        parser: IContentParser, 
        storage: IStorageBackend
    ):
        self.client = client
        self.parser = parser
        self.storage = storage
        
    def fetch(self, source_identifier: str) -> any:
        """Fetches the page data from Confluence."""
        return self.client.get_page(source_identifier)
        
    def parse(self, raw_data: any) -> str:
        """Parses the Confluence page data into Markdown."""
        title = raw_data.get('title', 'Untitled')
        
        # Extract HTML content
        body = raw_data.get('body', {})
        storage_format = body.get('storage', {})
        html_content = storage_format.get('value', '')
        
        parsed_content = self.parser.parse(html_content)
        
        # Add a title header to the markdown
        return f"# {title}\n\n{parsed_content}"
        
    def save(self, parsed_data: str, output_destination: str) -> None:
        """Saves the parsed markdown into the storage backend."""
        self.storage.save(parsed_data, output_destination)

    def extract_tree(self, root_page_id: str, output_dir: str) -> None:
        """
        Recursively extracts a page and all its child pages.
        """
        import os
        
        logger.info(f"Extracting tree starting from {root_page_id}")
        
        try:
            # 1. Determine destination and extract current page
            os.makedirs(output_dir, exist_ok=True)
            
            # Using inherited helper function sanitize_filename
            safe_filename = self.sanitize_filename(f"{root_page_id}.md")
            file_path = os.path.join(output_dir, safe_filename)
            
            # Use the Template Method 'extract' for the current item
            self.extract(source_identifier=root_page_id, output_destination=file_path)
            
            # Using inherited helper function log_success
            self.log_success(f"Successfully processed {root_page_id}")
            
            # 2. Fetch children and recursively extract them
            children = self.client.get_child_pages(root_page_id)
            if children:
                logger.info(f"Found {len(children)} child pages for {root_page_id}. Extracting...")
                for child in children:
                    child_id = child.get('id')
                    if child_id:
                        self.extract_tree(child_id, output_dir)
                        
        except Exception as e:
            logger.error(f"Failed to fetch or extract tree for page {root_page_id}: {e}")
