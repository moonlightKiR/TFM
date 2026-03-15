import requests
from typing import Dict, Any, List
from .interfaces import IConfluenceClient
from .config import ConfluenceConfig

class RequestsConfluenceClient(IConfluenceClient):
    """Concrete implementation of the API client utilizing requests."""
    
    def __init__(self, config: ConfluenceConfig):
        self.config = config
        self.session = requests.Session()
        self.session.auth = (self.config.username, self.config.api_token)
        self.session.headers.update({"Accept": "application/json"})
        
    def get_page(self, page_id: str) -> Dict[str, Any]:
        url = f"{self.config.url}/rest/api/content/{page_id}?expand=body.storage,version"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
        
    def get_child_pages(self, page_id: str) -> List[Dict[str, Any]]:
        url = f"{self.config.url}/rest/api/content/{page_id}/child/page"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json().get('results', [])
