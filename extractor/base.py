from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
    """
    Abstract base class for all data extractors.
    Allows creating new extractors for different sources (Confluence, Notion, Web Scraping, etc.)
    following the Template Method Pattern.
    """
    
    def extract(self, source_identifier: str, output_destination: str) -> None:
        """
        Executes the extraction process using a defined template: fetch -> parse -> save.
        
        :param source_identifier: The ID or URL to extract from.
        :param output_destination: Where to save the output (e.g., a file path).
        """
        logger.info(f"Starting extraction for source: {source_identifier}")
        raw_data = self.fetch(source_identifier)
        
        logger.info(f"Parsing extracted data...")
        parsed_data = self.parse(raw_data)
        
        logger.info(f"Saving data to: {output_destination}")
        self.save(parsed_data, output_destination)
        
        logger.info("Extraction completed successfully.")
        
    @abstractmethod
    def fetch(self, source_identifier: str) -> any:
        """Fetches raw data from the source."""
        pass
        
    @abstractmethod
    def parse(self, raw_data: any) -> str:
        """Parses raw data into the desired format."""
        pass
        
    @abstractmethod
    def save(self, parsed_data: str, output_destination: str) -> None:
        """Saves the parsed data to the destination."""
        pass
        
    def log_success(self, message: str) -> None:
        """Helper method inherited by all extractors to log success explicitly."""
        logger.info(f"[SUCCESS] {message}")
        
    def sanitize_filename(self, filename: str) -> str:
        """Helper method inherited by all extractors to clean illegal characters from filenames."""
        import re
        return re.sub(r'[\\/*?:"<>|]', "", filename)
