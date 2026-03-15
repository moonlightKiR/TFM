from bs4 import BeautifulSoup
from markdownify import markdownify as md
from .interfaces import IContentParser

class HtmlToMarkdownParser(IContentParser):
    """Concrete implementation parsing content into minimal markdown strings."""
    
    def parse(self, raw_content: str) -> str:
        if not raw_content:
            return ""
            
        # Parse cleanly avoiding potentially broken tags
        soup = BeautifulSoup(raw_content, 'html.parser')
        
        # Optionally perform cleanups on 'soup' before markdownizing
        
        # Convert to markdown
        markdown_text = md(str(soup), heading_style="ATX")
        return markdown_text.strip()
